"""Rendering the training corpus into the trainer's JSONL.

The property under test is the one the module exists for: **a training example
is the same bytes the evaluator would send for the same turn.** Everything else
here is downstream of that.

So the prompt is compared against ``render_example`` rather than against a
string typed out again, and the tools field is compared against the exact
``json.dumps`` call ``NeedleEngine`` makes. A test that restated either would
pass while the two paths drifted apart, which is the failure it is supposed to
catch.
"""

import json
import tempfile
import unittest
from pathlib import Path

from vesta_router.corpus import Case, load_corpus
from vesta_router.engines import tool_definitions
from vesta_router.render import render_example
from vesta_router.schema import load_tool_schema
from vesta_router.training import needle_example, render_training_examples, write_training_jsonl

REPO = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO / "tools" / "tool-schema-v2.json"
SCHEMA = load_tool_schema(SCHEMA_PATH)
TOOLS = tool_definitions(SCHEMA_PATH)
SYSTEM = "You control a phone."


def case(case_id="c1", utterance="Set a timer.", outcome=None, state=None) -> Case:
    return Case(
        id=case_id,
        utterance=utterance,
        locale="en",
        outcome=outcome or {"kind": "chat"},
        state=state,
        tags=(),
        source="test",
        category="test",
        file="test.jsonl",
        line=1,
    )


def render(c: Case) -> dict:
    return needle_example(c, SCHEMA, json.dumps(TOOLS, ensure_ascii=False), SYSTEM)


class PromptParityTests(unittest.TestCase):
    def test_the_query_is_the_evaluator_renderer_and_not_a_restatement(self):
        state = {
            "pending": {
                "tool": "set_reminder",
                "known": {"text": "call the dentist"},
                "missing": ["datetime"],
            }
        }
        c = case(
            utterance="at 4pm",
            state=state,
            outcome={
                "kind": "tool",
                "tool": "set_reminder",
                "arguments": {"text": "call the dentist", "datetime": "16:00"},
            },
        )
        self.assertEqual(render(c)["query"], render_example(c.utterance, c.state, SCHEMA))

    def test_the_tools_field_is_a_string_the_trainer_will_embed_verbatim(self):
        # needle's trainer re-serializes a list with compact separators and
        # embeds a string as given. NeedleEngine sends the spaced form, so the
        # string is what keeps training and inference on the same bytes.
        tools = render(case())["tools"]
        self.assertIsInstance(tools, str)
        self.assertEqual(tools, json.dumps(TOOLS, ensure_ascii=False))
        self.assertNotEqual(tools, json.dumps(TOOLS, ensure_ascii=False, separators=(",", ":")))


class OutcomeMappingTests(unittest.TestCase):
    def test_a_tool_outcome_is_one_call_with_its_arguments(self):
        c = case(outcome={"kind": "tool", "tool": "set_timer", "arguments": {"minutes": 20}})
        self.assertEqual(render(c)["answers"], [{"name": "set_timer", "arguments": {"minutes": 20}}])

    def test_incomplete_cancel_and_chat_all_teach_the_empty_answer(self):
        # The engine has no token for "I would call this but a slot is empty",
        # so `incomplete` is taught as withholding. That satisfies the outcome
        # assertion — no executable action reaches dispatch — and it is the
        # only satisfying behaviour the model can be trained toward.
        for outcome in (
            {"kind": "incomplete", "tool": "set_timer", "missing": ["minutes"]},
            {"kind": "cancel", "tool": "set_timer"},
            {"kind": "chat"},
        ):
            with self.subTest(kind=outcome["kind"]):
                example = render(case(outcome=outcome))
                self.assertEqual(example["answers"], [])
                self.assertEqual(example["outcome"], outcome["kind"])

    def test_no_reasoning_field_is_emitted(self):
        # A `reasoning` string would be trained as a <think> block. The corpus
        # authors none, so none is invented here.
        self.assertNotIn("reasoning", render(case()))


class DeterminismTests(unittest.TestCase):
    def test_the_same_corpus_renders_to_the_same_bytes(self):
        cases = load_corpus(REPO / "data" / "train")[0]
        with tempfile.TemporaryDirectory() as directory:
            first = write_training_jsonl(Path(directory) / "a.jsonl", cases, SCHEMA, TOOLS, SYSTEM)
            second = write_training_jsonl(Path(directory) / "b.jsonl", cases, SCHEMA, TOOLS, SYSTEM)
        self.assertEqual(first["sha256"], second["sha256"])
        self.assertEqual(first["examples"], len(cases))

    def test_the_summary_counts_outcomes_and_matches_the_corpus(self):
        cases = load_corpus(REPO / "data" / "train")[0]
        with tempfile.TemporaryDirectory() as directory:
            summary = write_training_jsonl(Path(directory) / "t.jsonl", cases, SCHEMA, TOOLS, SYSTEM)
        expected: dict[str, int] = {}
        for c in cases:
            expected[c.outcome["kind"]] = expected.get(c.outcome["kind"], 0) + 1
        self.assertEqual(summary["byOutcome"], expected)

    def test_examples_keep_corpus_order(self):
        cases = [case("a"), case("b"), case("c")]
        rendered = render_training_examples(cases, SCHEMA, TOOLS, SYSTEM)
        self.assertEqual([example["id"] for example in rendered], ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
