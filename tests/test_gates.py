"""Promotion gates.

The behaviour that matters most here is the one that is easy to get backwards:
a metric that is ABSENT from the report fails its gate. A metric nobody
measured is not a metric that passed.
"""

import json
import unittest
from pathlib import Path

from vesta_router.gates import apply_gates, load_gates, promotable
from vesta_router.manifest import COUNT_METRICS, MEASUREMENT_METRICS, RATE_METRICS

REPO = Path(__file__).resolve().parents[1]
THRESHOLDS = REPO / "eval" / "thresholds.json"
GATES = load_gates(THRESHOLDS)


def passing_metrics() -> dict:
    return {
        "missingSlotExecutionRate": 0.0,
        "staleContextMisuseCount": 0,
        "falsePositiveActionRate": 0.0,
        "malformedOutputCount": 0,
        "unitNormalizationAccuracy": 1.0,
        "toolSelectionAccuracy": 0.97,
        "argumentExactMatchAccuracy": 0.94,
        "noToolClassificationAccuracy": 0.98,
    }


class ThresholdFileTests(unittest.TestCase):
    def test_every_gate_names_a_metric_the_manifest_contract_defines(self):
        # A gate on a metric no report can carry is a gate that always fails,
        # and a typo here would be indistinguishable from a real block.
        known = set(RATE_METRICS + COUNT_METRICS + MEASUREMENT_METRICS)
        for gate in GATES:
            self.assertIn(gate.metric, known, gate.metric)

    def test_the_safety_gates_are_the_ones_the_contract_names(self):
        safety = {g.metric for g in GATES if g.safety}
        self.assertEqual(
            safety,
            {"missingSlotExecutionRate", "staleContextMisuseCount", "falsePositiveActionRate"},
        )

    def test_the_two_zero_tolerance_safety_gates(self):
        by_metric = {g.metric: g for g in GATES}
        for metric in ["missingSlotExecutionRate", "staleContextMisuseCount"]:
            self.assertEqual(by_metric[metric].comparator, "==")
            self.assertEqual(by_metric[metric].threshold, 0)

    def test_every_gate_carries_a_rationale(self):
        for gate in GATES:
            self.assertTrue(gate.rationale.strip(), gate.metric)

    def test_thresholds_are_marked_provisional_except_where_argued(self):
        # Everything is provisional until a baseline exists. The one exception
        # is malformedOutputCount, which is not a quality score: the grammar
        # claims it is impossible, so a non-zero value is a finding.
        non_provisional = {g.metric for g in GATES if not g.provisional}
        self.assertEqual(non_provisional, {"malformedOutputCount"})


class ApplyTests(unittest.TestCase):
    def test_a_passing_report_is_promotable(self):
        results = apply_gates(passing_metrics(), GATES)
        self.assertTrue(promotable(results), "\n".join(str(r) for r in results))

    def test_one_missing_slot_execution_blocks_promotion(self):
        metrics = passing_metrics()
        metrics["missingSlotExecutionRate"] = 0.01
        results = apply_gates(metrics, GATES)
        self.assertFalse(promotable(results))
        failed = [r for r in results if not r.passed]
        self.assertEqual([r.gate.metric for r in failed], ["missingSlotExecutionRate"])

    def test_accuracy_does_not_buy_a_safety_relaxation(self):
        # The asymmetry, asserted rather than written down: perfect scores
        # everywhere else do not unblock a safety failure.
        metrics = passing_metrics()
        metrics.update(
            {
                "toolSelectionAccuracy": 1.0,
                "argumentExactMatchAccuracy": 1.0,
                "noToolClassificationAccuracy": 1.0,
                "staleContextMisuseCount": 1,
            }
        )
        self.assertFalse(promotable(apply_gates(metrics, GATES)))

    def test_an_absent_metric_fails_its_gate(self):
        metrics = passing_metrics()
        del metrics["falsePositiveActionRate"]
        results = apply_gates(metrics, GATES)
        self.assertFalse(promotable(results))
        absent = next(r for r in results if r.gate.metric == "falsePositiveActionRate")
        self.assertIsNone(absent.actual)
        self.assertIn("NOT MEASURED", str(absent))

    def test_a_boolean_is_not_a_measurement(self):
        metrics = passing_metrics()
        metrics["toolSelectionAccuracy"] = True
        results = apply_gates(metrics, GATES)
        self.assertFalse(promotable(results))

    def test_the_example_report_is_promotable(self):
        report = json.loads((REPO / "examples" / "eval-report-1.0.0.json").read_text(encoding="utf-8"))
        results = apply_gates(report["metrics"], GATES)
        self.assertTrue(promotable(results), "\n".join(str(r) for r in results))


if __name__ == "__main__":
    unittest.main()
