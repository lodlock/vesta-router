"""The case-level severity diff, and several runs compared as one table.

The five classification rules are asserted one at a time, because the one that
matters is not the one that fires most often. ``get_time`` becoming
``navigate_to`` on the same sentence leaves every existing number identical —
same family, same count, same ``falsePositiveActionRate`` — and it is a real
safety regression. If that case passes and the others fail, the model is still
doing its job; if that case fails, nothing else here is worth much.

The case documents are the shape ``cases.jsonl`` actually carries, including
the ``safety`` block evaluator 1.1.0 writes, so a change to the report's
serialization breaks these tests rather than quietly making the diff read
nothing.
"""

import json
import unittest

from vesta_router.failures import case_safety, compare, compare_summaries, diff_safety


def scored(case_id="c1", kind="chat", calls=(), severity=None, guarded=None, families_from=None):
    """One scored case line, with its severity block.

    ``calls`` and ``severity`` are given independently on purpose: a test that
    derived the severity from the calls would be asserting the evaluator's
    behaviour a second time rather than the diff's.
    """
    executable = guarded == "executable_action" if guarded else severity is not None
    classification = {
        "expectsAction": kind == "tool",
        "correctTool": False,
        "wrongTool": False,
        "extraTool": False,
        "noToolWhenExpected": False,
        "toolWhenNoneExpected": kind == "chat" and bool(calls),
        "argumentsExact": False,
        "fullCallExact": False,
        "missingExpectedArguments": [],
        "unexpectedExtraArguments": [],
        "wrongValuedArguments": [],
        "executableAction": executable,
        "fabricatedRequiredValue": False,
        "placeholderRequiredValue": False,
        "staleContextMisuse": [],
        "unitNormalizationCorrect": None,
        "resetOk": True,
        "stateLeak": False,
        "nondeterministic": False,
    }
    if families_from:
        classification.update(families_from)
    return {
        "id": case_id,
        "utterance": "thanks",
        "expected": {"kind": kind, "tool": None, "arguments": None, "missing": []},
        "rawOutcome": "tool_call" if calls else "no_call",
        "guardedOutcome": guarded or ("executable_action" if executable else "no_executable_action"),
        "engine": {"functionCalls": [{"name": n, "arguments": {}} for n in calls]},
        "screened": [],
        "classification": classification,
        "safety": {
            "falseActions": [{"tool": n, "severity": severity, "classified": True} for n in calls]
            if severity
            else [],
            "highestFalseActionSeverity": severity,
            "falseActionWeight": 0,
            "unclassifiedTools": [],
        },
        "provenance": {
            "artifactSha256": "a" * 64,
            "baselineId": "baseline-x-aaaaaaaaaaaa",
            "corpusContentSha256": "b" * 64,
            "engineVersion": "3.0.1",
            "evaluatorVersion": "1.1.0",
            "toolSchemaVersion": 2,
        },
    }


def without_safety(case):
    """The same line as evaluator 1.0.0 wrote it: no severity block at all."""
    stripped = json.loads(json.dumps(case))
    stripped.pop("safety")
    return stripped


class DiffSafetyRules(unittest.TestCase):
    def _diff(self, before_severity, after_severity, before_tools=("get_time",), after_tools=("get_time",)):
        before = case_safety(scored(calls=before_tools if before_severity else (), severity=before_severity))
        after = case_safety(scored(calls=after_tools if after_severity else (), severity=after_severity))
        return diff_safety(before, after)

    def test_a_removed_false_action_is_an_improvement(self):
        self.assertEqual(self._diff("high_impact", None)["verdict"], "improved")

    def test_a_lower_severity_replacement_is_an_improvement(self):
        result = self._diff("high_impact", "read_only", ("make_call",), ("get_time",))
        self.assertEqual(result["verdict"], "improved")
        self.assertIn("fell from high_impact to read_only", result["reason"])

    def test_a_higher_severity_replacement_is_a_regression(self):
        result = self._diff("read_only", "user_visible_side_effect", ("get_time",), ("navigate_to",))
        self.assertEqual(result["verdict"], "regressed")
        self.assertIn("rose from read_only to user_visible_side_effect", result["reason"])
        self.assertIn("get_time -> navigate_to", result["reason"])

    def test_a_correct_outcome_becoming_a_false_action_is_a_regression(self):
        self.assertEqual(self._diff(None, "read_only")["verdict"], "regressed")

    def test_neither_side_acting_is_unchanged(self):
        result = self._diff(None, None)
        self.assertEqual(result["verdict"], "unchanged")
        self.assertEqual(result["reason"], "no false action on either side")

    def test_the_same_false_action_is_unchanged(self):
        self.assertEqual(self._diff("read_only", "read_only")["verdict"], "unchanged")

    def test_a_tool_swap_at_the_same_severity_is_unchanged_but_named(self):
        # Not a safety movement, and still a behavioural one. Dropping it would
        # make the diff look quieter than the run was.
        result = self._diff("read_only", "read_only", ("get_time",), ("search_contacts",))
        self.assertEqual(result["verdict"], "unchanged")
        self.assertIn("get_time -> search_contacts", result["reason"])

    def test_a_missing_severity_block_is_unavailable_not_unchanged(self):
        # An absent severity is not an absence of severe failures.
        self.assertEqual(diff_safety(None, case_safety(scored()))["verdict"], "unavailable")
        self.assertEqual(diff_safety(case_safety(scored()), None)["verdict"], "unavailable")
        self.assertEqual(diff_safety(None, None)["verdict"], "unavailable")

    def test_the_rules_are_deterministic(self):
        for _ in range(3):
            self.assertEqual(
                self._diff("read_only", "high_impact"),
                self._diff("read_only", "high_impact"),
            )


class CompareCarriesTheSeverityDiff(unittest.TestCase):
    def test_a_severity_rise_alone_makes_the_comparison_a_safety_regression(self):
        """The case the boolean was blind to, end to end.

        Same family, same executable action, same false-positive rate — and the
        thing that would have run went from reading the clock to starting
        turn-by-turn navigation.
        """
        before = [scored("neg.typo", calls=("get_time",), severity="read_only")]
        after = [scored("neg.typo", calls=("navigate_to",), severity="user_visible_side_effect")]

        document = compare(before, after)

        # Nothing moved in the family view.
        self.assertEqual(document["improved"], [])
        self.assertEqual(document["regressed"], [])
        self.assertEqual(document["safetyRegressions"], [])
        self.assertEqual([d for d in document["familyDeltas"] if d["delta"]], [])

        # And the verdict is still a safety regression.
        self.assertEqual(document["verdict"], "safety-regression")
        self.assertEqual(document["safetyRegressionCount"], 1)
        self.assertEqual(document["safetySeverityRegressions"], ["neg.typo"])

    def test_the_case_level_diff_records_both_sides(self):
        before = [scored("c1", calls=("make_call",), severity="high_impact")]
        after = [scored("c1", calls=("get_time",), severity="read_only")]
        entry = compare(before, after)["caseSafetyDiff"][0]
        self.assertEqual(entry["id"], "c1")
        self.assertEqual(entry["expectedKind"], "chat")
        self.assertEqual(entry["beforeSeverity"], "high_impact")
        self.assertEqual(entry["afterSeverity"], "read_only")
        self.assertEqual(entry["before"]["tools"], ["make_call"])
        self.assertEqual(entry["after"]["tools"], ["get_time"])
        self.assertEqual(entry["verdict"], "improved")

    def test_unchanged_cases_stay_out_of_the_case_level_diff(self):
        same = [scored("c1", calls=("get_time",), severity="read_only")]
        self.assertEqual(compare(same, same)["caseSafetyDiff"], [])

    def test_counts_are_reported_from_the_case_diff(self):
        before = [
            scored("a", calls=("get_time",), severity="read_only"),
            scored("b", calls=("make_call",), severity="high_impact"),
        ]
        after = [
            scored("a", calls=("make_call",), severity="high_impact"),
            scored("b", calls=(), severity=None),
        ]
        document = compare(before, after)
        self.assertEqual(document["safetyRegressionCount"], 1)
        self.assertEqual(document["safetyImprovementCount"], 1)

    def test_the_weighted_delta_is_null_when_either_side_predates_the_model(self):
        before = [without_safety(scored("c1", calls=("get_time",)))]
        after = [scored("c1", calls=("get_time",), severity="read_only")]
        document = compare(before, after)
        self.assertIsNone(document["weightedFalseActionScoreDelta"])
        self.assertEqual(document["safetySeverityUnavailableCount"], 1)
        self.assertFalse(document["baseline"]["severity"]["available"])
        self.assertTrue(document["candidate"]["severity"]["available"])

    def test_an_unavailable_severity_is_not_counted_as_a_regression(self):
        before = [without_safety(scored("c1", calls=("make_call",)))]
        after = [scored("c1", calls=("make_call",), severity="high_impact")]
        document = compare(before, after)
        self.assertEqual(document["safetyRegressionCount"], 0)
        self.assertEqual(document["safetyImprovementCount"], 0)


class ThreeWayComparison(unittest.TestCase):
    """Three summaries in one table — the shape that can attribute a delta."""

    def summary(self, label, metrics, evaluator="1.1.0", corpus="b" * 64, cases=64):
        return {
            "baselineId": label,
            "artifactSha256": "a" * 64,
            "evaluatorVersion": evaluator,
            "corpusContentSha256": corpus,
            "toolSchemaVersion": 2,
            "cases": cases,
            "partialRun": False,
            "metrics": metrics,
            "performance": {"medianLatencyMs": 200.0},
        }

    def test_a_metric_the_control_matches_is_attributed_away_from_training(self):
        # The whole point: published differs, control and candidate agree, so
        # the delta belongs to the build path rather than to the fine-tune.
        document = compare_summaries(
            [
                self.summary("published", {"toolSelectionAccuracy": 0.964286}),
                self.summary("control", {"toolSelectionAccuracy": 0.928571}),
                self.summary("candidate", {"toolSelectionAccuracy": 0.928571}),
            ]
        )
        row = document["metrics"][0]
        self.assertFalse(row["identical"])
        self.assertEqual(row["values"]["control"], row["values"]["candidate"])
        self.assertNotEqual(row["values"]["published"], row["values"]["control"])

    def test_identical_and_differing_metrics_are_listed_separately(self):
        document = compare_summaries(
            [
                self.summary("a", {"same": 1, "moved": 1}),
                self.summary("b", {"same": 1, "moved": 2}),
                self.summary("c", {"same": 1, "moved": 2}),
            ]
        )
        self.assertEqual(document["metricsIdenticalAcrossRuns"], ["same"])
        self.assertEqual(document["metricsDiffering"], ["moved"])

    def test_a_metric_missing_from_one_run_is_null_and_not_zero(self):
        document = compare_summaries(
            [
                self.summary("old", {}, evaluator="1.0.0"),
                self.summary("new", {"weightedFalseActionScore": 193}),
            ]
        )
        row = next(r for r in document["metrics"] if r["metric"] == "weightedFalseActionScore")
        self.assertIsNone(row["values"]["old"])
        self.assertFalse(row["presentInEveryRun"])

    def test_a_different_evaluator_makes_the_table_not_comparable(self):
        document = compare_summaries(
            [self.summary("a", {}, evaluator="1.0.0"), self.summary("b", {}, evaluator="1.1.0")]
        )
        self.assertFalse(document["comparable"])
        self.assertEqual([p["field"] for p in document["incomparabilities"]], ["evaluatorVersion"])

    def test_a_different_corpus_makes_the_table_not_comparable(self):
        document = compare_summaries([self.summary("a", {}), self.summary("b", {}, corpus="c" * 64)])
        self.assertFalse(document["comparable"])
        self.assertIn("corpusContentSha256", [p["field"] for p in document["incomparabilities"]])

    def test_a_different_case_count_is_named(self):
        document = compare_summaries([self.summary("a", {}), self.summary("b", {}, cases=32)])
        self.assertFalse(document["comparable"])
        self.assertIn("cases", [p["field"] for p in document["incomparabilities"]])

    def test_matched_runs_are_comparable(self):
        document = compare_summaries([self.summary("a", {}), self.summary("b", {})])
        self.assertTrue(document["comparable"])
        self.assertEqual(document["incomparabilities"], [])

    def test_dict_valued_metrics_compare_by_value(self):
        histogram = {"read_only": 6, "high_impact": 3}
        document = compare_summaries(
            [
                self.summary("a", {"falseActionCountBySeverity": dict(histogram)}),
                self.summary("b", {"falseActionCountBySeverity": dict(reversed(list(histogram.items())))}),
            ]
        )
        # Key order must not make two equal histograms look different.
        self.assertEqual(document["metricsIdenticalAcrossRuns"], ["falseActionCountBySeverity"])


class StableSerialization(unittest.TestCase):
    """Two runs of the same analysis must diff to nothing."""

    def _cases(self):
        return [
            scored("a", calls=("get_time",), severity="read_only"),
            scored("b", calls=("make_call",), severity="high_impact"),
            scored("c", calls=(), severity=None),
        ]

    def _dump(self, document):
        return json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False)

    def test_compare_serializes_identically_on_repeat(self):
        before, after = self._cases(), self._cases()
        self.assertEqual(self._dump(compare(before, after)), self._dump(compare(before, after)))

    def test_compare_does_not_depend_on_input_order(self):
        before, after = self._cases(), self._cases()
        shuffled = list(reversed(after))
        self.assertEqual(self._dump(compare(before, after)), self._dump(compare(before, shuffled)))

    def test_the_case_level_diff_is_ordered_by_case_id(self):
        before = [
            scored("z", calls=("make_call",), severity="high_impact"),
            scored("a", calls=("get_time",), severity="read_only"),
        ]
        after = [scored("z", calls=(), severity=None), scored("a", calls=(), severity=None)]
        ids = [entry["id"] for entry in compare(before, after)["caseSafetyDiff"]]
        self.assertEqual(ids, sorted(ids))

    def test_compare_summaries_serializes_identically_on_repeat(self):
        runs = [
            ThreeWayComparison().summary("a", {"m": 1}),
            ThreeWayComparison().summary("b", {"m": 2}),
        ]
        self.assertEqual(self._dump(compare_summaries(runs)), self._dump(compare_summaries(runs)))

    def test_every_document_survives_a_json_round_trip(self):
        # Nothing here may carry a tuple, a set or a dataclass: a report that
        # cannot be re-read is not machine-readable.
        document = compare(self._cases(), self._cases())
        self.assertEqual(json.loads(self._dump(document)), json.loads(self._dump(document)))


if __name__ == "__main__":
    unittest.main()
