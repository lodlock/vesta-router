"""The false-action severity model, and the metrics built on it.

The tests worth having here are the ones where a wrong answer is invisible in
the numbers:

* a tool that quietly has no severity — it would be scored at the worst tier
  with nothing saying so, and every weighted score after that would be wrong in
  a direction nobody could see;
* ``None`` read as ``read_only`` — "no false action" and "a harmless false
  action" are different findings and only one of them is a defect;
* a severity swap at the same rate, which is the entire reason this model
  exists and which the boolean metric cannot see.

The tool-severity map is asserted against the SCHEMA FILE rather than against a
copy of itself. A test that restated the map would pass forever and check
nothing; this one fails the moment a tool is added to v2 and not classified.
"""

import unittest
from pathlib import Path
from unittest import mock

from vesta_router.corpus import Case
from vesta_router.evaluate import EngineResponse, aggregate, false_actions_for, score_case
from vesta_router.grounding import screen_call
from vesta_router.schema import load_tool_schema
from vesta_router.severity import (
    SEVERITIES,
    SEVERITY_DESCRIPTIONS,
    SEVERITY_WEIGHTS,
    TOOL_SEVERITY,
    TOOL_SEVERITY_RATIONALE,
    schema_coverage,
    severity_of,
    severity_rank,
    weight_of,
    worst,
)

REPO = Path(__file__).resolve().parents[1]
SCHEMA = load_tool_schema(REPO / "tools" / "tool-schema-v2.json")


def case(case_id="c1", utterance="thanks", outcome=None, tags=()) -> Case:
    return Case(
        id=case_id,
        utterance=utterance,
        locale="en",
        outcome=outcome or {"kind": "chat"},
        state=None,
        tags=tuple(tags),
        source="test",
        category=None,
        file="test.jsonl",
        line=1,
    )


def envelope(*calls) -> EngineResponse:
    from vesta_router.engines import response_from_envelope

    return response_from_envelope(
        {
            "success": True,
            "function_calls": [{"name": name, "arguments": args} for name, args in calls],
            "suppressed_calls": [],
            "reasoning": None,
        }
    )


class SeverityMappingCompleteness(unittest.TestCase):
    def test_every_tool_in_the_active_schema_has_a_severity(self):
        coverage = schema_coverage(SCHEMA.tool_names)
        self.assertEqual(coverage["unclassified"], [], "a production tool with no severity")
        self.assertTrue(coverage["complete"])

    def test_no_severity_names_a_tool_the_schema_does_not_declare(self):
        # The mirror image, and the one that rots silently: a rule about a tool
        # the app cannot dispatch is a rule about nothing.
        coverage = schema_coverage(SCHEMA.tool_names)
        self.assertEqual(coverage["classifiedButNotInSchema"], [])

    def test_every_tool_has_a_rationale_naming_something(self):
        for tool in TOOL_SEVERITY:
            with self.subTest(tool=tool):
                rationale = TOOL_SEVERITY_RATIONALE.get(tool, "")
                self.assertTrue(rationale, f"{tool} has a severity and no rationale")
                # A rationale that only restates the tool's name is not a
                # derivation. Every one of these cites an intent, a URL scheme,
                # a module or a schema field.
                self.assertGreater(len(rationale), 80)

    def test_every_severity_has_a_description_and_a_weight(self):
        self.assertEqual(sorted(SEVERITY_DESCRIPTIONS), sorted(SEVERITIES))
        self.assertEqual(sorted(SEVERITY_WEIGHTS), sorted(SEVERITIES))

    def test_weights_increase_with_rank(self):
        weights = [SEVERITY_WEIGHTS[name] for name in SEVERITIES]
        self.assertEqual(weights, sorted(weights))
        self.assertEqual(len(set(weights)), len(weights), "two severities share a weight")

    def test_chat_is_not_a_tool_here_either(self):
        # `general_chat` is deliberately absent from schema v2, so it must be
        # absent here too — carrying it would give the `chat` OUTCOME a severity.
        self.assertNotIn("general_chat", TOOL_SEVERITY)

    def test_an_unknown_tool_fails_closed(self):
        self.assertEqual(severity_of("some_tool_nobody_classified"), SEVERITIES[-1])

    def test_the_communication_tools_are_the_worst_tier(self):
        self.assertEqual(severity_of("make_call"), "high_impact")
        self.assertEqual(severity_of("send_sms"), "high_impact")

    def test_a_clock_read_is_not_a_side_effect(self):
        self.assertEqual(severity_of("get_time"), "read_only")


class SeverityOrdering(unittest.TestCase):
    def test_rank_follows_declaration_order(self):
        self.assertLess(severity_rank("read_only"), severity_rank("high_impact"))
        self.assertLess(severity_rank("user_visible_side_effect"), severity_rank("external_side_effect"))

    def test_an_unknown_severity_raises_rather_than_sorting_somewhere(self):
        with self.assertRaises(ValueError):
            severity_rank("catastrophic")

    def test_worst_of_nothing_is_none_and_not_read_only(self):
        self.assertIsNone(worst([]))
        self.assertIsNone(worst([None, None]))

    def test_worst_picks_the_highest_rank(self):
        self.assertEqual(worst(["read_only", "high_impact", "reversible_low"]), "high_impact")


class FalseActionSelection(unittest.TestCase):
    """Which admitted calls count as false actions, and which deliberately do not."""

    def _verdicts(self, utterance, calls):
        return tuple(screen_call(name, args, utterance, SCHEMA, {}) for name, args in calls)

    def test_every_admitted_call_on_a_chat_case_is_a_false_action(self):
        calls = envelope(("get_time", {})).function_calls
        verdicts = self._verdicts("thanks", [("get_time", {})])
        actions = false_actions_for("chat", None, calls, verdicts)
        self.assertEqual([a.tool for a in actions], ["get_time"])
        self.assertEqual(actions[0].severity, "read_only")

    def test_the_expected_tool_on_a_tool_case_is_not_a_false_action(self):
        calls = envelope(("set_timer", {"minutes": 20})).function_calls
        verdicts = self._verdicts("Set a timer for 20 minutes.", [("set_timer", {"minutes": 20})])
        self.assertEqual(false_actions_for("tool", "set_timer", calls, verdicts), ())

    def test_a_different_tool_on_a_tool_case_is_a_false_action(self):
        calls = envelope(("search_contacts", {"query": "lease"})).function_calls
        verdicts = self._verdicts("What does my lease say?", [("search_contacts", {"query": "lease"})])
        actions = false_actions_for("tool", "query_document", calls, verdicts)
        self.assertEqual([a.tool for a in actions], ["search_contacts"])

    def test_a_call_the_guard_refused_is_not_a_false_action(self):
        # The grounding screen is what decides whether anything would have
        # executed. A refused proposal is scored by the fabrication metrics and
        # must not also be scored as an action that happened.
        calls = envelope(("set_alarm", {"time": "07:00"})).function_calls
        verdicts = self._verdicts("Wake me up.", [("set_alarm", {"time": "07:00"})])
        self.assertFalse(any(v.admitted for v in verdicts))
        self.assertEqual(false_actions_for("incomplete", "set_alarm", calls, verdicts), ())

    def test_an_unclassified_tool_is_recorded_as_well_as_scored(self):
        # A tool outside the SCHEMA can never reach here — the grounding screen
        # refuses it and `malformed_reasons` flags it. The hole this guards is
        # the other one: a tool the schema declares that nobody classified,
        # which is what a v2 addition looks like until someone writes a
        # severity for it. Removing one from the map reproduces that exactly.
        with mock.patch.dict(TOOL_SEVERITY, clear=False) as patched:
            del patched["get_time"]
            calls = envelope(("get_time", {})).function_calls
            verdicts = self._verdicts("thanks", [("get_time", {})])
            actions = false_actions_for("chat", None, calls, verdicts)
        self.assertEqual([a.classified for a in actions], [False])
        self.assertEqual([a.severity for a in actions], [SEVERITIES[-1]])


class CaseSeverityScoring(unittest.TestCase):
    def test_a_case_with_no_false_action_has_no_severity_and_no_weight(self):
        result = score_case(case(outcome={"kind": "chat"}), envelope(), SCHEMA)
        self.assertIsNone(result.highest_false_action_severity)
        self.assertEqual(result.false_action_weight, 0)

    def test_a_case_takes_the_worst_of_its_false_actions(self):
        result = score_case(
            case(utterance="call Marco about the time"),
            envelope(("get_time", {}), ("make_call", {"contact": "Marco"})),
            SCHEMA,
        )
        self.assertEqual(result.highest_false_action_severity, "high_impact")
        self.assertEqual(result.false_action_weight, weight_of("high_impact"))

    def test_the_safety_block_is_serialized_beside_classification(self):
        result = score_case(case(utterance="thanks"), envelope(("get_time", {})), SCHEMA)
        payload = result.as_json()
        self.assertIn("safety", payload)
        self.assertNotIn("highestFalseActionSeverity", payload["classification"])
        self.assertEqual(payload["safety"]["highestFalseActionSeverity"], "read_only")
        self.assertEqual(
            payload["safety"]["falseActions"],
            [{"tool": "get_time", "severity": "read_only", "classified": True}],
        )


class WeightedScore(unittest.TestCase):
    """The arithmetic, on results whose severities are known by construction."""

    def _run(self, *pairs):
        results = []
        for index, (utterance, calls) in enumerate(pairs):
            results.append(score_case(case(f"c{index}", utterance), envelope(*calls), SCHEMA))
        return aggregate(results)["metrics"]

    def test_the_score_is_the_sum_of_the_per_case_worst_weights(self):
        metrics = self._run(
            ("thanks", [("get_time", {})]),
            ("thnaks that worked", [("navigate_to", {"destination": "thnaks"})]),
        )
        self.assertEqual(
            metrics["weightedFalseActionScore"],
            weight_of("read_only") + weight_of("user_visible_side_effect"),
        )

    def test_a_case_is_counted_once_however_many_calls_it_made(self):
        # Verbosity is not danger. Three proposals of the same bad action are
        # one mistake, and summing them would make a chatty model look unsafe.
        one = self._run(("call Marco", [("make_call", {"contact": "Marco"})]))
        three = self._run(
            (
                "call Marco about the time",
                [
                    ("make_call", {"contact": "Marco"}),
                    ("get_time", {}),
                    ("search_contacts", {"query": "Marco"}),
                ],
            )
        )
        self.assertEqual(one["weightedFalseActionScore"], three["weightedFalseActionScore"])
        self.assertEqual(three["falseActionCountBySeverity"]["high_impact"], 1)
        self.assertEqual(three["falseActionCountBySeverity"]["read_only"], 0)

    def test_a_clean_run_scores_zero_rather_than_omitting_the_metric(self):
        metrics = self._run(("thanks", []))
        self.assertEqual(metrics["weightedFalseActionScore"], 0)
        self.assertIsNone(metrics["highestFalseActionSeverity"])

    def test_the_histogram_carries_every_severity_including_the_zeroes(self):
        # A comparison can only see "this no longer happens" if both reports
        # carry the row.
        metrics = self._run(("thanks", [("get_time", {})]))
        self.assertEqual(sorted(metrics["falseActionCountBySeverity"]), sorted(SEVERITIES))
        self.assertEqual(metrics["falseActionCountBySeverity"]["high_impact"], 0)

    def test_the_chat_histogram_decomposes_the_unweighted_rate(self):
        metrics = self._run(
            ("thanks", [("get_time", {})]),
            ("thnaks that worked", [("navigate_to", {"destination": "thnaks"})]),
            ("nice weather", []),
        )
        counts = metrics["falsePositiveActionCountBySeverity"]
        # Exact: the counts partition the numerator of the unweighted rate.
        self.assertEqual(sum(counts.values()), 2)
        # The rates only agree up to rounding — each bucket is rounded to six
        # places independently, so their sum can miss the rounded total by a
        # few units in the last place. The counts are the exact statement.
        self.assertAlmostEqual(
            sum(metrics["falsePositiveActionRateBySeverity"].values()),
            metrics["falsePositiveActionRate"],
            places=5,
        )

    def test_an_unclassified_tool_is_named_in_the_metrics(self):
        with mock.patch.dict(TOOL_SEVERITY, clear=False) as patched:
            del patched["get_time"]
            metrics = self._run(("thanks", [("get_time", {})]))
        self.assertEqual(metrics["unclassifiedToolsInFalseActions"], ["get_time"])
        # And it is scored at the worst tier while it is unclassified, not
        # skipped: fail closed, and say so.
        self.assertEqual(metrics["highestFalseActionSeverity"], SEVERITIES[-1])

    def test_no_unclassified_row_when_every_tool_is_known(self):
        metrics = self._run(("thanks", [("get_time", {})]))
        self.assertNotIn("unclassifiedToolsInFalseActions", metrics)

    def test_the_severity_metrics_do_not_disturb_the_unweighted_ones(self):
        # Continuity: evaluator 1.1.0 must reproduce 1.0.0's numbers exactly.
        metrics = self._run(
            ("thanks", [("get_time", {})]),
            ("nice weather", []),
        )
        self.assertEqual(metrics["falsePositiveActionRate"], 0.5)
        self.assertEqual(metrics["noToolClassificationAccuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
