"""Running the corpus: reset policy, state isolation, and the report on disk.

None of this loads a model. The engine here is a scripted stand-in, which is
the only way to test the thing that matters most — that a leak between cases is
DETECTED — because a real engine that leaks on demand is not something you can
ask for.
"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from vesta_router.corpus import Case
from vesta_router.engines import response_from_envelope
from vesta_router.evaluate import EngineResponse
from vesta_router.runner import (
    EVALUATOR_VERSION,
    baseline_identifier,
    corpus_digest,
    run_evaluation,
    write_report,
)
from vesta_router.schema import load_tool_schema

REPO = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO / "tools" / "tool-schema-v2.json"
SCHEMA = load_tool_schema(SCHEMA_PATH)


def envelope(calls=()):
    return {
        "type": "call",
        "success": True,
        "function_calls": [{"name": n, "arguments": a} for n, a in calls],
        "suppressed_calls": [],
        "reasoning": "because",
        "prefill_tps": 900.0,
        "decode_tps": 380.0,
        "peak_ram_mb": 112.4,
    }


def case(case_id, utterance, outcome, state=None, tags=()):
    return Case(
        id=case_id,
        utterance=utterance,
        locale="en",
        outcome=outcome,
        state=state,
        tags=tuple(tags),
        source="test",
        category=None,
        file="test.jsonl",
        line=1,
    )


TIMER = {"kind": "tool", "tool": "set_timer", "arguments": {"minutes": 20}}
ALARM = {"kind": "tool", "tool": "set_alarm", "arguments": {"time": "07:30"}}

CASES = [
    case("a", "Set a timer for 20 minutes.", TIMER),
    case("b", "Wake me at 7:30 tomorrow.", ALARM),
]


class ScriptedEngine:
    """Answers from a script, and counts what the runner asked it to do."""

    instances: list["ScriptedEngine"] = []

    def __init__(self, script, fail_reset_on=()):
        # script: prompt -> list of envelopes, consumed in order across the
        # whole life of ONE engine instance.
        self.script = {prompt: list(answers) for prompt, answers in script.items()}
        self.fail_reset_on = set(fail_reset_on)
        self.resets = 0
        self.prompts: list[str] = []
        self.closed = False
        ScriptedEngine.instances.append(self)

    def describe(self):
        return {
            "engine": "scripted",
            "engineVersion": "0",
            "engineLibrary": None,
            "artifactPath": "/tmp/fake.cact",
            "artifactSha256": "0" * 64,
            "artifactSizeBytes": 1,
            "artifactKind": "cact",
            "systemPrompt": "test",
            "autoDate": False,
        }

    def reset(self):
        self.resets += 1
        return self.resets not in self.fail_reset_on

    def complete(self, prompt):
        self.prompts.append(prompt)
        queue = self.script.get(prompt)
        if not queue:
            return EngineResponse(error=f"no script for {prompt!r}")
        body = queue.pop(0) if len(queue) > 1 else queue[0]
        return response_from_envelope(body, 10.0)

    def close(self):
        self.closed = True


def factory_for(script, fail_reset_on=()):
    def make():
        return ScriptedEngine(script, fail_reset_on)

    return make


class ResetPolicyTests(unittest.TestCase):
    def setUp(self):
        ScriptedEngine.instances.clear()

    def test_every_case_is_preceded_by_a_reset(self):
        script = {
            "Set a timer for 20 minutes.": [envelope([("set_timer", {"minutes": 20})])],
            "Wake me at 7:30 tomorrow.": [envelope([("set_alarm", {"time": "07:30"})])],
        }
        outcome = run_evaluation(factory_for(script), CASES, SCHEMA, isolation_check=False)
        self.assertEqual(ScriptedEngine.instances[0].resets, 2)
        self.assertTrue(ScriptedEngine.instances[0].closed)
        self.assertTrue(all(r.reset_ok for r in outcome.results))

    def test_a_failed_reset_is_recorded_against_the_case_that_followed(self):
        script = {
            "Set a timer for 20 minutes.": [envelope([("set_timer", {"minutes": 20})])],
            "Wake me at 7:30 tomorrow.": [envelope([("set_alarm", {"time": "07:30"})])],
        }
        outcome = run_evaluation(factory_for(script, fail_reset_on={2}), CASES, SCHEMA, isolation_check=False)
        self.assertTrue(outcome.results[0].reset_ok)
        self.assertFalse(outcome.results[1].reset_ok)

    def test_the_engine_is_closed_even_when_a_case_raises(self):
        class Exploding(ScriptedEngine):
            def complete(self, prompt):
                raise RuntimeError("boom")

        with self.assertRaises(RuntimeError):
            run_evaluation(lambda: Exploding({}), CASES, SCHEMA, isolation_check=False)
        self.assertTrue(ScriptedEngine.instances[-1].closed)


class IsolationTests(unittest.TestCase):
    def setUp(self):
        ScriptedEngine.instances.clear()

    def test_a_clean_run_reports_no_leak(self):
        script = {
            "Set a timer for 20 minutes.": [envelope([("set_timer", {"minutes": 20})])],
            "Wake me at 7:30 tomorrow.": [envelope([("set_alarm", {"time": "07:30"})])],
        }
        outcome = run_evaluation(factory_for(script), CASES, SCHEMA, isolation_check=True)
        self.assertTrue(outcome.isolation_ran)
        self.assertEqual([r.state_leak for r in outcome.results], [False, False])
        self.assertEqual([r.nondeterministic for r in outcome.results], [False, False])
        # A fresh engine for the isolation pass, not the one already used.
        self.assertEqual(len(ScriptedEngine.instances), 2)

    def test_an_answer_that_depends_on_what_ran_before_is_a_leak(self):
        # The isolation engine answers differently, twice consistently: the
        # case's answer depended on its neighbours, which is exactly the
        # failure reset is supposed to prevent.
        pass_a = {
            "Set a timer for 20 minutes.": [envelope([("set_timer", {"minutes": 20})])],
            "Wake me at 7:30 tomorrow.": [envelope([("set_alarm", {"time": "07:30"})])],
        }
        pass_b = {
            "Set a timer for 20 minutes.": [envelope([("set_timer", {"minutes": 20})])],
            "Wake me at 7:30 tomorrow.": [envelope([("set_alarm", {"time": "06:00"})])],
        }
        scripts = [pass_a, pass_b]

        def make():
            return ScriptedEngine(scripts[len(ScriptedEngine.instances)])

        outcome = run_evaluation(make, CASES, SCHEMA, isolation_check=True)
        self.assertEqual([r.state_leak for r in outcome.results], [False, True])
        self.assertEqual([r.nondeterministic for r in outcome.results], [False, False])

    def test_two_different_answers_in_a_row_is_nondeterminism_not_a_leak(self):
        # No leak claim is made about an engine that cannot reproduce itself.
        pass_a = {
            "Set a timer for 20 minutes.": [envelope([("set_timer", {"minutes": 20})])],
            "Wake me at 7:30 tomorrow.": [envelope([("set_alarm", {"time": "07:30"})])],
        }
        pass_b = {
            "Set a timer for 20 minutes.": [envelope([("set_timer", {"minutes": 20})])],
            "Wake me at 7:30 tomorrow.": [
                envelope([("set_alarm", {"time": "06:00"})]),
                envelope([("set_alarm", {"time": "05:00"})]),
            ],
        }
        scripts = [pass_a, pass_b]

        def make():
            return ScriptedEngine(scripts[len(ScriptedEngine.instances)])

        outcome = run_evaluation(make, CASES, SCHEMA, isolation_check=True)
        self.assertEqual([r.nondeterministic for r in outcome.results], [False, True])
        self.assertEqual([r.state_leak for r in outcome.results], [False, False])

    def test_metrics_come_from_the_scored_pass_only(self):
        # Pass B answers the timer wrongly. The metric must not move.
        pass_a = {"Set a timer for 20 minutes.": [envelope([("set_timer", {"minutes": 20})])]}
        pass_b = {"Set a timer for 20 minutes.": [envelope([("set_timer", {"minutes": 2})])]}
        scripts = [pass_a, pass_b]

        def make():
            return ScriptedEngine(scripts[len(ScriptedEngine.instances)])

        outcome = run_evaluation(make, CASES[:1], SCHEMA, isolation_check=True)
        self.assertTrue(outcome.results[0].full_call_exact)
        self.assertTrue(outcome.results[0].state_leak)

    def test_skipping_the_isolation_pass_is_visible_in_the_outcome(self):
        script = {"Set a timer for 20 minutes.": [envelope([("set_timer", {"minutes": 20})])]}
        outcome = run_evaluation(factory_for(script), CASES[:1], SCHEMA, isolation_check=False)
        self.assertFalse(outcome.isolation_ran)
        self.assertEqual(len(ScriptedEngine.instances), 1)


class IdentityTests(unittest.TestCase):
    def test_a_baseline_id_names_a_file_not_a_version(self):
        digest = "c9d915eca282ed42d1a09b143b592adb4cc6744ffe2d294adf5cfc5548170c38"
        self.assertEqual(
            baseline_identifier("/models/needle3.cact", digest),
            "baseline-needle3-c9d915eca282",
        )

    def test_it_is_not_a_parseable_model_version(self):
        from vesta_router.version import parse_model_version

        digest = "a" * 64
        self.assertIsNone(parse_model_version(baseline_identifier("x.cact", digest)))

    def test_the_corpus_digest_is_content_not_order(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "b.jsonl").write_bytes(b"second\n")
            (root / "a.jsonl").write_bytes(b"first\n")
            forwards = corpus_digest([root / "a.jsonl", root / "b.jsonl"])
            backwards = corpus_digest([root / "b.jsonl", root / "a.jsonl"])
            self.assertEqual(forwards, backwards)
            (root / "a.jsonl").write_bytes(b"changed\n")
            self.assertNotEqual(corpus_digest([root / "a.jsonl", root / "b.jsonl"]), forwards)


class ReportTests(unittest.TestCase):
    def setUp(self):
        ScriptedEngine.instances.clear()
        script = {
            "Set a timer for 20 minutes.": [envelope([("set_timer", {"minutes": 20})])],
            "Wake me at 7:30 tomorrow.": [envelope([("set_alarm", {"time": "07:30"})])],
        }
        self.outcome = run_evaluation(factory_for(script), CASES, SCHEMA, isolation_check=False)

    def _write(self, root, notes=()):
        return write_report(
            root,
            self.outcome,
            SCHEMA,
            SCHEMA_PATH,
            sorted((REPO / "data" / "eval").glob("*.jsonl")),
            "baseline-test-000000000000",
            notes=notes,
        )

    def test_the_three_files(self):
        with TemporaryDirectory() as tmp:
            directory = self._write(tmp)
            self.assertTrue((directory / "summary.json").is_file())
            self.assertTrue((directory / "manifest.json").is_file())
            self.assertTrue((directory / "cases.jsonl").is_file())

    def test_one_line_per_case_and_each_one_is_attributable(self):
        with TemporaryDirectory() as tmp:
            directory = self._write(tmp)
            lines = (directory / "cases.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), len(CASES))
            first = json.loads(lines[0])
            self.assertEqual(first["provenance"]["artifactSha256"], "0" * 64)
            self.assertEqual(first["provenance"]["evaluatorVersion"], EVALUATOR_VERSION)
            self.assertEqual(first["provenance"]["toolSchemaVersion"], 2)
            self.assertIn("renderedPrompt", first)
            self.assertIn("rawEnvelope", first["engine"])

    def test_serialization_is_deterministic(self):
        with TemporaryDirectory() as tmp:
            one = json.loads((self._write(tmp) / "summary.json").read_text(encoding="utf-8"))
            two = json.loads((self._write(tmp) / "summary.json").read_text(encoding="utf-8"))
            for payload in (one, two):
                payload.pop("evaluatedAt")
                payload["performance"].pop("totalWallClockMs", None)
            self.assertEqual(one, two)

    def test_keys_are_sorted_so_a_diff_is_a_change(self):
        with TemporaryDirectory() as tmp:
            text = (self._write(tmp) / "summary.json").read_text(encoding="utf-8")
            payload = json.loads(text)
            self.assertEqual(text, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

    def test_the_report_says_what_it_is_and_what_it_is_not(self):
        with TemporaryDirectory() as tmp:
            directory = self._write(tmp, notes=("a note",))
            summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["kind"], "baseline")
            self.assertIsNone(summary["modelVersion"])
            self.assertIn("a note", summary["notes"])
            self.assertTrue(manifest["policy"]["resetBetweenCases"])
            self.assertFalse(manifest["policy"]["isolationPass"])
            self.assertIn("NOT the app's own code", manifest["policy"]["guardrail"])

    def test_a_baseline_summary_is_gateable(self):
        # The gates are what a future candidate is judged against, so the
        # baseline must be measurable by exactly the same reader. An untrained
        # model is expected to be BLOCKED; what matters is that it is blocked
        # on numbers rather than on a shape the reader cannot parse.
        from vesta_router.gates import apply_gates, load_gates

        with TemporaryDirectory() as tmp:
            summary = json.loads((self._write(tmp) / "summary.json").read_text(encoding="utf-8"))
            results = apply_gates(summary["metrics"], load_gates(REPO / "eval" / "thresholds.json"))
            self.assertTrue(any(r.actual is not None for r in results))


if __name__ == "__main__":
    unittest.main()
