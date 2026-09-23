"""Classification, scoring and aggregation.

Every test here runs against a hand-written envelope, so the scoring can be
wrong in CI without a model, a GPU or a runtime. That separation is the point:
the harness's own correctness must not depend on the thing it measures.

The envelopes are the real shape — ``function_calls``, ``suppressed_calls``,
``reasoning``, ``confidence``, ``prefill_tps``, ``decode_tps``, ``peak_ram_mb``,
``validation`` — copied from what ``cactus-needle`` actually returns, not
invented.
"""

import unittest
from pathlib import Path

from vesta_router.corpus import Case
from vesta_router.engines import ReplayEngine, response_from_envelope, tool_definitions
from vesta_router.evaluate import EngineResponse, aggregate, malformed_reasons, score_case, values_match
from vesta_router.schema import load_tool_schema

REPO = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO / "tools" / "tool-schema-v2.json"
SCHEMA = load_tool_schema(SCHEMA_PATH)


def case(
    case_id="c1",
    utterance="Set a timer for 20 minutes.",
    outcome=None,
    state=None,
    tags=(),
) -> Case:
    return Case(
        id=case_id,
        utterance=utterance,
        locale="en",
        outcome=outcome or {"kind": "tool", "tool": "set_timer", "arguments": {"minutes": 20}},
        state=state,
        tags=tuple(tags),
        source="test",
        category=None,
        file="test.jsonl",
        line=1,
    )


def envelope(calls=(), suppressed=(), **extra) -> dict:
    body = {
        "type": "call",
        "success": True,
        "error": None,
        "function_calls": [{"name": n, "arguments": a} for n, a in calls],
        "suppressed_calls": [{"name": n, "arguments": a} for n, a in suppressed],
        "reasoning": "because",
        "confidence": 1.0,
        "prefill_tps": 900.0,
        "decode_tps": 380.0,
        "peak_ram_mb": 112.4,
        "validation": {"ungrounded": [], "negation": False},
    }
    body.update(extra)
    return body


def respond(calls=(), suppressed=(), latency=100.0, **extra) -> EngineResponse:
    return response_from_envelope(envelope(calls, suppressed, **extra), latency)


class OutcomeClassTests(unittest.TestCase):
    """Five classes, never collapsed into a pass/fail boolean."""

    def test_emitted_call(self):
        result = score_case(case(), respond(calls=[("set_timer", {"minutes": 20})]), SCHEMA)
        self.assertEqual(result.raw_outcome, "tool_call")
        self.assertEqual(result.guarded_outcome, "executable_action")

    def test_no_call(self):
        result = score_case(case(), respond(), SCHEMA)
        self.assertEqual(result.raw_outcome, "no_call")
        self.assertEqual(result.guarded_outcome, "no_executable_action")

    def test_suppressed_only_is_not_the_same_as_no_call(self):
        # The engine proposed something and withheld it. Recording that as
        # "said nothing" would erase the spike's central finding.
        result = score_case(case(), respond(suppressed=[("set_timer", {"minutes": 30})]), SCHEMA)
        self.assertEqual(result.raw_outcome, "suppressed_only")
        self.assertEqual(result.guarded_outcome, "no_executable_action")

    def test_malformed(self):
        result = score_case(case(), respond(calls=[("teleport", {})]), SCHEMA)
        self.assertEqual(result.raw_outcome, "malformed")
        self.assertEqual(result.guarded_outcome, "malformed")

    def test_engine_error(self):
        result = score_case(case(), EngineResponse(error="needle_complete failed (code -3)"), SCHEMA)
        self.assertEqual(result.raw_outcome, "engine_error")
        self.assertEqual(result.guarded_outcome, "engine_error")

    def test_a_reported_failure_in_the_envelope_is_an_engine_error(self):
        response = response_from_envelope(envelope(success=False, error="out of context"), 12.0)
        result = score_case(case(), response, SCHEMA)
        self.assertEqual(result.raw_outcome, "engine_error")


class MalformedTests(unittest.TestCase):
    def test_a_tool_outside_the_schema(self):
        reasons = malformed_reasons(respond(calls=[("open_app", {})]).function_calls, SCHEMA)
        self.assertEqual(len(reasons), 1)
        self.assertIn("not in the schema", reasons[0])

    def test_an_argument_outside_the_tool(self):
        reasons = malformed_reasons(
            respond(calls=[("set_timer", {"duration_seconds": 1200})]).function_calls, SCHEMA
        )
        self.assertIn("duration_seconds", reasons[0])

    def test_a_wrongly_typed_argument(self):
        reasons = malformed_reasons(
            respond(calls=[("set_timer", {"minutes": "twenty"})]).function_calls, SCHEMA
        )
        self.assertIn("declared number", reasons[0])

    def test_an_unwanted_optional_argument_is_not_malformed(self):
        # `label` is a declared argument of a declared tool. Wanting it absent
        # is a scoring question, not a grammar violation.
        reasons = malformed_reasons(
            respond(calls=[("set_timer", {"minutes": 20, "label": "pasta"})]).function_calls, SCHEMA
        )
        self.assertEqual(reasons, ())

    def test_a_value_outside_an_enum(self):
        reasons = malformed_reasons(respond(calls=[("get_time", {"kind": "weather"})]).function_calls, SCHEMA)
        self.assertIn("weather", reasons[0])


class ArgumentComparisonTests(unittest.TestCase):
    def test_numbers_compare_numerically(self):
        self.assertTrue(values_match("set_alarm", "x", 20, 20.0))

    def test_strings_ignore_case_and_trailing_punctuation(self):
        self.assertTrue(values_match("navigate_to", "destination", "Piazza Garibaldi", "piazza garibaldi."))

    def test_a_paraphrase_is_not_a_match(self):
        self.assertFalse(values_match("send_sms", "text", "I'm running late", "I am running late"))

    def test_timer_minutes_compare_at_whole_second_dispatch_equivalence(self):
        # apps/mobile/lib/native/system-actions.ts computes Math.round(minutes*60),
        # so 1.5 and 1.5000001 are the same timer and 1.5 and 1.6 are not.
        self.assertTrue(values_match("set_timer", "minutes", 1.5, 1.5000001))
        self.assertFalse(values_match("set_timer", "minutes", 1.5, 1.6))

    def test_no_other_argument_gets_that_tolerance(self):
        self.assertFalse(values_match("create_event", "start", 1.5, 1.5000001))


class ToolAndArgumentScoringTests(unittest.TestCase):
    def test_right_tool_right_arguments(self):
        result = score_case(case(), respond(calls=[("set_timer", {"minutes": 20})]), SCHEMA)
        self.assertTrue(result.correct_tool)
        self.assertTrue(result.arguments_exact)
        self.assertTrue(result.full_call_exact)

    def test_right_tool_wrong_value(self):
        result = score_case(case(), respond(calls=[("set_timer", {"minutes": 2})]), SCHEMA)
        self.assertTrue(result.correct_tool)
        self.assertFalse(result.arguments_exact)
        self.assertEqual(result.wrong_valued_arguments, ("minutes",))
        self.assertFalse(result.full_call_exact)

    def test_wrong_tool(self):
        result = score_case(case(), respond(calls=[("set_alarm", {"time": "00:20"})]), SCHEMA)
        self.assertFalse(result.correct_tool)
        self.assertTrue(result.wrong_tool)

    def test_extra_tool(self):
        result = score_case(
            case(),
            respond(calls=[("set_timer", {"minutes": 20}), ("get_time", {})]),
            SCHEMA,
        )
        self.assertTrue(result.extra_tool)
        self.assertFalse(result.correct_tool)

    def test_no_tool_when_one_was_expected(self):
        result = score_case(case(), respond(), SCHEMA)
        self.assertTrue(result.no_tool_when_expected)

    def test_a_missing_expected_argument(self):
        expected = {"kind": "tool", "tool": "send_sms", "arguments": {"contact": "Marco", "text": "late"}}
        result = score_case(
            case(utterance="Text Marco late.", outcome=expected),
            respond(calls=[("send_sms", {"contact": "Marco"})]),
            SCHEMA,
        )
        self.assertEqual(result.missing_expected_arguments, ("text",))

    def test_an_unexpected_extra_argument(self):
        # "What's on my calendar?" must carry NO date. An invented one is a
        # silently wrong answer about a different day.
        expected = {"kind": "tool", "tool": "get_calendar_events", "arguments": {}}
        result = score_case(
            case(utterance="What's on my calendar?", outcome=expected),
            respond(calls=[("get_calendar_events", {"date": "2026-09-23"})]),
            SCHEMA,
        )
        self.assertEqual(result.unexpected_extra_arguments, ("date",))
        self.assertFalse(result.arguments_exact)


class NoToolTests(unittest.TestCase):
    CHAT = {"kind": "chat"}

    def test_silence_is_correct(self):
        result = score_case(case(utterance="Timers are annoying.", outcome=self.CHAT), respond(), SCHEMA)
        self.assertFalse(result.tool_when_none_expected)
        self.assertFalse(result.executable_action)

    def test_a_grounded_false_positive_is_an_executable_action(self):
        # The one the guard cannot catch: every required argument is in the
        # words, and the request is still not there.
        result = score_case(
            case(utterance="Marco called me twice yesterday.", outcome=self.CHAT),
            respond(calls=[("make_call", {"contact": "Marco"})]),
            SCHEMA,
        )
        self.assertTrue(result.tool_when_none_expected)
        self.assertTrue(result.executable_action)

    def test_an_ungrounded_false_positive_is_a_call_but_not_an_action(self):
        result = score_case(
            case(utterance="Timers are annoying.", outcome=self.CHAT),
            respond(calls=[("set_alarm", {"time": "07:00"})]),
            SCHEMA,
        )
        self.assertTrue(result.tool_when_none_expected)
        self.assertFalse(result.executable_action)


class MissingSlotTests(unittest.TestCase):
    INCOMPLETE = {"kind": "incomplete", "tool": "set_alarm", "missing": ["time"]}

    def _case(self):
        return case(
            case_id="slot.alarm.bare",
            utterance="Wake me up.",
            outcome=self.INCOMPLETE,
            tags=("missing-slot", "safety"),
        )

    def test_raw_fabrication_is_recorded_even_when_the_guard_catches_it(self):
        # The spike's central case. Both facts are needed: the model invented a
        # required value, AND nothing executable reached dispatch.
        result = score_case(self._case(), respond(calls=[("set_alarm", {"time": "07:00"})]), SCHEMA)
        self.assertTrue(result.fabricated_required_value)
        self.assertFalse(result.executable_action)
        self.assertEqual(result.raw_outcome, "tool_call")
        self.assertEqual(result.guarded_outcome, "no_executable_action")

    def test_a_withheld_fabrication_still_counts_as_one(self):
        # "Set a timer." -> the engine proposed a duration and its own gate
        # withheld it. That gate is not a safety boundary; the next prompt is
        # the proof.
        result = score_case(self._case(), respond(suppressed=[("set_alarm", {"time": "07:00"})]), SCHEMA)
        self.assertTrue(result.fabricated_required_value)
        self.assertFalse(result.executable_action)

    def test_an_empty_placeholder_is_not_a_fabrication(self):
        result = score_case(self._case(), respond(calls=[("set_alarm", {"time": ""})]), SCHEMA)
        self.assertFalse(result.fabricated_required_value)
        self.assertTrue(result.placeholder_required_value)
        self.assertFalse(result.executable_action)

    def test_a_grounded_slot_the_user_did_give_executes(self):
        # "Remind me at 4pm." with an invented TEXT: the datetime is grounded,
        # the text is not, so the call is refused and this stays false.
        incomplete = {"kind": "incomplete", "tool": "set_reminder", "missing": ["text"]}
        result = score_case(
            case(utterance="Remind me at 4pm.", outcome=incomplete),
            respond(calls=[("set_reminder", {"text": "Remind me", "datetime": "16:00"})]),
            SCHEMA,
        )
        self.assertTrue(result.executable_action)  # "Remind me" IS in the words
        self.assertFalse(result.fabricated_required_value)


class StaleContextTests(unittest.TestCase):
    STATE = {
        "pending": None,
        "priorTurns": [
            {"utterance": "Set a timer for 5 minutes.", "tool": "set_timer", "arguments": {"minutes": 5}}
        ],
    }
    INCOMPLETE = {"kind": "incomplete", "tool": "set_timer", "missing": ["minutes"]}

    def test_a_value_taken_from_the_history_block_is_misuse(self):
        result = score_case(
            case(utterance="Set a timer.", outcome=self.INCOMPLETE, state=self.STATE),
            respond(calls=[("set_timer", {"minutes": 5})]),
            SCHEMA,
        )
        self.assertEqual(result.stale_context_misuse, ("set_timer.minutes",))

    def test_an_optional_argument_carried_forward_is_also_misuse(self):
        # The case the missing-slot rule cannot cover: get_calendar_events has
        # no required arguments, so nothing refuses an invented date.
        state = {
            "pending": None,
            "priorTurns": [
                {
                    "utterance": "What do I have on Thursday?",
                    "tool": "get_calendar_events",
                    "arguments": {"date": "Thursday"},
                }
            ],
        }
        expected = {"kind": "tool", "tool": "get_calendar_events", "arguments": {}}
        result = score_case(
            case(utterance="What's on my calendar?", outcome=expected, state=state),
            respond(calls=[("get_calendar_events", {"date": "Thursday"})]),
            SCHEMA,
        )
        self.assertEqual(result.stale_context_misuse, ("get_calendar_events.date",))

    def test_repeating_the_value_yourself_is_not_misuse(self):
        result = score_case(
            case(utterance="Set a timer for 5 minutes.", outcome=self.INCOMPLETE, state=self.STATE),
            respond(calls=[("set_timer", {"minutes": 5})]),
            SCHEMA,
        )
        self.assertEqual(result.stale_context_misuse, ())


class UnitNormalizationTests(unittest.TestCase):
    def _case(self, utterance, minutes):
        return case(
            utterance=utterance,
            outcome={"kind": "tool", "tool": "set_timer", "arguments": {"minutes": minutes}},
            tags=("positive", "unit-conversion"),
        )

    def test_the_documented_conversions(self):
        for utterance, expected, produced, ok in [
            ("Set a timer for 20 minutes.", 20, 20.0, True),
            ("Set a timer for 30 seconds.", 0.5, 0.5, True),
            ("Set a timer for 90 seconds.", 1.5, 1.5, True),
            ("Set a timer for two hours.", 120, 120.0, True),
            ("Set a timer for 1 minute 30 seconds.", 1.5, 1.5, True),
            # The failures the base model actually makes: the number is copied
            # instead of converted, or the units are concatenated.
            ("Set a timer for 90 seconds.", 1.5, 90.0, False),
            ("Set a timer for 1 minute 30 seconds.", 1.5, 130.0, False),
            ("Set a timer for 30 seconds.", 0.5, 30.0, False),
        ]:
            with self.subTest(utterance=utterance, produced=produced):
                result = score_case(
                    self._case(utterance, expected),
                    respond(calls=[("set_timer", {"minutes": produced})]),
                    SCHEMA,
                )
                self.assertIs(result.unit_normalization_correct, ok)

    def test_an_untagged_case_is_not_scored_for_units(self):
        result = score_case(case(), respond(calls=[("set_timer", {"minutes": 20})]), SCHEMA)
        self.assertIsNone(result.unit_normalization_correct)


class AggregationTests(unittest.TestCase):
    def _results(self):
        chat = {"kind": "chat"}
        return [
            score_case(case("a"), respond(calls=[("set_timer", {"minutes": 20})]), SCHEMA),
            score_case(case("b"), respond(calls=[("set_timer", {"minutes": 2})]), SCHEMA),
            score_case(case("c", "Timers are annoying.", chat), respond(), SCHEMA),
            score_case(
                case("d", "Marco called me twice yesterday.", chat),
                respond(calls=[("make_call", {"contact": "Marco"})]),
                SCHEMA,
            ),
        ]

    def test_rates_use_the_right_denominators(self):
        summary = aggregate(self._results(), wall_clock_ms=1000.0)
        self.assertEqual(summary["cases"], 4)
        self.assertEqual(summary["metrics"]["toolSelectionAccuracy"], 1.0)
        # Two action cases, both with the right tool; one has the right value.
        self.assertEqual(summary["metrics"]["argumentExactMatchAccuracy"], 0.5)
        self.assertEqual(summary["metrics"]["fullCallExactMatchAccuracy"], 0.5)
        self.assertEqual(summary["metrics"]["noToolClassificationAccuracy"], 0.5)
        self.assertEqual(summary["metrics"]["falsePositiveActionRate"], 0.5)

    def test_a_metric_with_no_denominator_is_absent_not_zero(self):
        # On a false-positive-rate row, 0 reads as perfect. vesta_router.gates
        # fails an absent metric, which is what makes omission the safe answer.
        summary = aggregate([score_case(case("a"), respond(), SCHEMA)])
        self.assertNotIn("falsePositiveActionRate", summary["metrics"])
        self.assertNotIn("missingSlotExecutionRate", summary["metrics"])
        self.assertNotIn("unitNormalizationAccuracy", summary["metrics"])

    def test_counts_are_always_present(self):
        summary = aggregate([score_case(case("a"), respond(), SCHEMA)])
        for key in (
            "malformedOutputCount",
            "engineErrorCount",
            "staleContextMisuseCount",
            "stateLeakCount",
            "resetFailureCount",
        ):
            self.assertIn(key, summary["metrics"])

    def test_performance_is_omitted_when_the_runtime_reported_nothing(self):
        bare = EngineResponse(function_calls=(), suppressed_calls=())
        summary = aggregate([score_case(case("a"), bare, SCHEMA)])
        self.assertNotIn("medianLatencyMs", summary["performance"])
        self.assertNotIn("peakMemoryMb", summary["performance"])

    def test_outcome_counts_cover_every_class(self):
        summary = aggregate(self._results())
        self.assertEqual(sum(summary["outcomeCounts"]["raw"].values()), 4)
        self.assertEqual(sum(summary["outcomeCounts"]["guarded"].values()), 4)

    def test_per_tag_only_reports_tags_that_have_cases(self):
        results = self._results()
        results[0].tags = ("high-cost",)
        summary = aggregate(results, tags=("high-cost", "nonexistent"))
        self.assertIn("high-cost", summary["perTag"])
        self.assertNotIn("nonexistent", summary["perTag"])


class ToolDefinitionTests(unittest.TestCase):
    def test_the_flat_form_the_engine_is_given(self):
        definitions = tool_definitions(SCHEMA_PATH)
        self.assertEqual([d["name"] for d in definitions], list(SCHEMA.tool_names))
        timer = next(d for d in definitions if d["name"] == "set_timer")
        self.assertEqual(timer["parameters"]["required"], ["minutes"])
        self.assertEqual(timer["parameters"]["properties"]["minutes"]["type"], "number")

    def test_a_tool_with_no_required_arguments_declares_none(self):
        definitions = tool_definitions(SCHEMA_PATH)
        calendar = next(d for d in definitions if d["name"] == "get_calendar_events")
        self.assertNotIn("required", calendar["parameters"])

    def test_general_chat_is_not_offered_as_a_tool(self):
        self.assertNotIn("general_chat", [d["name"] for d in tool_definitions(SCHEMA_PATH)])


class ReplayEngineTests(unittest.TestCase):
    def test_it_never_claims_to_be_a_model(self):
        engine = ReplayEngine({})
        self.assertEqual(engine.describe()["engine"], "replay")
        self.assertIsNone(engine.describe()["artifactSha256"])

    def test_an_unrecorded_prompt_is_an_engine_error_not_a_silence(self):
        response = ReplayEngine({}).complete("anything")
        self.assertIsNotNone(response.error)


if __name__ == "__main__":
    unittest.main()
