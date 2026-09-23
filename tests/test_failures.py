"""Failure-family classification and the two-run comparison.

Every case document here is the shape ``cases.jsonl`` actually carries — the
fields are copied from the real baseline report, not invented — so a change to
the report's serialization breaks these tests rather than silently making the
analysis read nothing.

The tests worth having are the ones where a wrong answer would be invisible:
a family that fires on the wrong denominator, a comparison that calls a swapped
failure an improvement, and the zero-count rows that are the only way a NEWLY
INTRODUCED failure can be seen at all.
"""

import unittest

from vesta_router.failures import FAMILIES, analyze, classify_case, compare


def scored(
    case_id="c1",
    kind="chat",
    raw_outcome="tool_call",
    guarded="executable_action",
    calls=(),
    screened=(),
    **classification,
):
    body = {
        "expectsAction": kind == "tool",
        "correctTool": False,
        "wrongTool": False,
        "extraTool": False,
        "noToolWhenExpected": False,
        "toolWhenNoneExpected": False,
        "argumentsExact": False,
        "fullCallExact": False,
        "missingExpectedArguments": [],
        "unexpectedExtraArguments": [],
        "wrongValuedArguments": [],
        "executableAction": guarded == "executable_action",
        "fabricatedRequiredValue": False,
        "placeholderRequiredValue": False,
        "staleContextMisuse": [],
        "unitNormalizationCorrect": None,
        "resetOk": True,
        "stateLeak": False,
        "nondeterministic": False,
    }
    body.update(classification)
    return {
        "id": case_id,
        "expected": {"kind": kind, "tool": None, "arguments": None, "missing": []},
        "rawOutcome": raw_outcome,
        "guardedOutcome": guarded,
        "engine": {"functionCalls": [{"name": n, "arguments": a} for n, a in calls]},
        "screened": list(screened),
        "classification": body,
        "provenance": {
            "artifactSha256": "a" * 64,
            "baselineId": "baseline-x-aaaaaaaaaaaa",
            "corpusContentSha256": "b" * 64,
            "engineVersion": "3.0.1",
            "evaluatorVersion": "1.0.0",
            "toolSchemaVersion": 2,
        },
    }


class FamilyTests(unittest.TestCase):
    def test_an_executable_action_on_a_chat_case_is_the_safety_family(self):
        families = classify_case(scored(kind="chat", toolWhenNoneExpected=True))
        self.assertIn("false-positive-executable-action", families)
        self.assertIn("false-positive-tool-call", families)

    def test_a_withheld_call_on_a_chat_case_is_not_a_false_positive_at_all(self):
        # suppressed_only: the engine proposed and withheld. Nothing was
        # emitted, so neither false-positive family applies.
        families = classify_case(
            scored(kind="chat", raw_outcome="suppressed_only", guarded="no_executable_action")
        )
        self.assertNotIn("false-positive-tool-call", families)
        self.assertNotIn("false-positive-executable-action", families)

    def test_families_respect_their_denominator(self):
        # An executable action is a failure on a `chat` case and the whole
        # point on a `tool` one. A family that ignored expected kind would
        # report every correct answer as a defect.
        self.assertEqual(classify_case(scored(kind="tool", correctTool=True, argumentsExact=True)), ())

    def test_correct_tool_wrong_arguments_needs_the_right_tool(self):
        wrong_args = scored(kind="tool", correctTool=True, argumentsExact=False)
        self.assertIn("correct-tool-wrong-arguments", classify_case(wrong_args))
        wrong_tool = scored(kind="tool", wrongTool=True)
        self.assertNotIn("correct-tool-wrong-arguments", classify_case(wrong_tool))
        self.assertIn("wrong-tool", classify_case(wrong_tool))

    def test_lexical_grounded_wrong_value_reads_the_screening_verdicts(self):
        # "Send Marco a text." -> text: "a text". Every required argument was
        # judged grounded, and the corpus says the case must not execute.
        admitted = scored(
            case_id="slot.sms.no-text",
            kind="incomplete",
            calls=(("send_sms", {"contact": "Marco", "text": "a text"}),),
            screened=[
                {
                    "tool": "send_sms",
                    "admitted": True,
                    "reason": None,
                    "requiredArguments": [
                        {"name": "contact", "state": "grounded", "evidence": "Marco"},
                        {"name": "text", "state": "grounded", "evidence": "a text"},
                    ],
                }
            ],
        )
        self.assertIn("lexical-grounded-wrong-value", classify_case(admitted))

    def test_a_refused_call_is_not_lexically_grounded(self):
        refused = scored(
            kind="incomplete",
            guarded="no_executable_action",
            screened=[
                {
                    "tool": "set_alarm",
                    "admitted": False,
                    "reason": "set_alarm.time has no evidence",
                    "requiredArguments": [{"name": "time", "state": "ungrounded", "evidence": None}],
                }
            ],
        )
        self.assertNotIn("lexical-grounded-wrong-value", classify_case(refused))

    def test_reset_failure_is_absent_rather_than_false_when_unrecorded(self):
        # A report that never recorded resetOk must not read as a reset that
        # failed — absent is not a value here either.
        case = scored()
        del case["classification"]["resetOk"]
        self.assertNotIn("reset-failure", classify_case(case))


class AnalyzeTests(unittest.TestCase):
    def test_every_family_is_reported_even_at_zero(self):
        document = analyze([scored(kind="tool", correctTool=True, argumentsExact=True)])
        self.assertEqual(len(document["families"]), len(FAMILIES))
        self.assertEqual(document["observedFamilies"], [])
        self.assertEqual(document["casesWithNoFamily"], ["c1"])

    def test_provenance_that_disagrees_between_cases_is_null_not_the_first_answer(self):
        other = scored(case_id="c2")
        other["provenance"]["artifactSha256"] = "c" * 64
        document = analyze([scored(), other])
        self.assertIsNone(document["provenance"]["artifactSha256"])
        self.assertEqual(document["provenance"]["evaluatorVersion"], "1.0.0")


class CompareTests(unittest.TestCase):
    BEFORE = [
        scored(case_id="fixed", kind="chat", toolWhenNoneExpected=True),
        scored(case_id="kept", kind="chat", toolWhenNoneExpected=True),
        scored(case_id="clean", kind="tool", guarded="no_executable_action"),
    ]

    def test_a_family_that_disappears_is_an_improvement(self):
        after = [
            scored(case_id="fixed", kind="chat", raw_outcome="no_call", guarded="no_executable_action"),
            scored(case_id="kept", kind="chat", toolWhenNoneExpected=True),
            scored(case_id="clean", kind="tool", guarded="no_executable_action"),
        ]
        result = compare(self.BEFORE, after)
        self.assertEqual([entry["id"] for entry in result["improved"]], ["fixed"])
        self.assertEqual([entry["id"] for entry in result["unchangedFailures"]], ["kept"])
        self.assertEqual(result["verdict"], "improved")

    def test_a_new_safety_family_is_named_a_safety_regression_whatever_else_improved(self):
        after = [
            scored(case_id="fixed", kind="chat", raw_outcome="no_call", guarded="no_executable_action"),
            scored(case_id="kept", kind="chat", toolWhenNoneExpected=True),
            scored(case_id="clean", kind="tool", guarded="no_executable_action", staleContextMisuse=["x.y"]),
        ]
        result = compare(self.BEFORE, after)
        self.assertEqual(result["safetyRegressions"], ["clean"])
        self.assertEqual(result["verdict"], "safety-regression")

    def test_swapping_one_failure_for_another_is_neither_improved_nor_regressed(self):
        after = [
            scored(case_id="fixed", kind="chat", toolWhenNoneExpected=True),
            scored(case_id="kept", kind="chat", toolWhenNoneExpected=True),
            scored(case_id="clean", kind="tool", guarded="no_executable_action", noToolWhenExpected=True),
        ]
        result = compare(self.BEFORE, after)
        self.assertEqual([entry["id"] for entry in result["regressed"]], ["clean"])

    def test_family_deltas_name_the_cases_on_both_sides(self):
        after = [
            scored(case_id="fixed", kind="chat", raw_outcome="no_call", guarded="no_executable_action"),
            scored(case_id="kept", kind="chat", toolWhenNoneExpected=True),
            scored(case_id="clean", kind="tool", guarded="no_executable_action"),
        ]
        deltas = {d["family"]: d for d in compare(self.BEFORE, after)["familyDeltas"]}
        entry = deltas["false-positive-tool-call"]
        self.assertEqual((entry["before"], entry["after"], entry["delta"]), (2, 1, -1))
        self.assertEqual(entry["fixed"], ["fixed"])
        self.assertEqual(entry["introduced"], [])

    def test_cases_present_on_only_one_side_are_reported_not_compared(self):
        result = compare(self.BEFORE, self.BEFORE[:2])
        self.assertEqual(result["onlyInBaseline"], ["clean"])
        self.assertEqual(result["casesComparedById"], 2)


if __name__ == "__main__":
    unittest.main()
