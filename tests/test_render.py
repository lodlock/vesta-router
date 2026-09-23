"""The state-block renderer.

Determinism is the property under test. The same state must always render to
the same bytes, or the model sees two prompts for one situation.
"""

import unittest
from pathlib import Path

from vesta_router.corpus import load_corpus
from vesta_router.render import render_example, render_state
from vesta_router.schema import load_tool_schema

REPO = Path(__file__).resolve().parents[1]
SCHEMA = load_tool_schema(REPO / "tools" / "tool-schema-v2.json")


class NoStateTests(unittest.TestCase):
    def test_none_renders_nothing_at_all(self):
        # Not an empty block. A stateless turn must produce the bytes a
        # stateless turn actually produces in the app.
        self.assertEqual(render_state(None, SCHEMA), "")
        self.assertEqual(render_example("20 minutes", None, SCHEMA), "20 minutes")

    def test_an_empty_state_object_also_renders_nothing(self):
        self.assertEqual(render_state({"pending": None, "priorTurns": []}, SCHEMA), "")


class PendingTests(unittest.TestCase):
    STATE = {"pending": {"tool": "set_timer", "known": {}, "missing": ["minutes"]}}

    def test_the_documented_shape(self):
        self.assertEqual(
            render_example("20 minutes", self.STATE, SCHEMA),
            "[pending]\ntool=set_timer\nknown={}\nmissing=minutes\n[/pending]\n\n20 minutes",
        )

    def test_known_keys_are_sorted(self):
        # The same state written in two orders must render identically, or the
        # corpus quietly contains two prompts for one situation.
        a = {"pending": {"tool": "send_sms", "known": {"contact": "Anna", "text": "hi"}, "missing": []}}
        b = {"pending": {"tool": "send_sms", "known": {"text": "hi", "contact": "Anna"}, "missing": []}}
        self.assertEqual(render_state(a, SCHEMA), render_state(b, SCHEMA))
        self.assertIn('known={"contact":"Anna","text":"hi"}', render_state(a, SCHEMA))

    def test_missing_is_in_schema_order_not_file_order(self):
        typed_backwards = {"pending": {"tool": "set_reminder", "known": {}, "missing": ["datetime", "text"]}}
        # set_reminder declares text before datetime.
        self.assertIn("missing=text,datetime", render_state(typed_backwards, SCHEMA))

    def test_non_ascii_is_not_escaped(self):
        state = {"pending": {"tool": "navigate_to", "known": {"destination": "Città Studi"}, "missing": []}}
        self.assertIn("Città Studi", render_state(state, SCHEMA))


class HistoryTests(unittest.TestCase):
    STATE = {
        "pending": None,
        "priorTurns": [
            {"utterance": "Ring Anna please.", "tool": "make_call", "arguments": {"contact": "Anna"}}
        ],
    }

    def test_prior_turns_render_as_completed_actions(self):
        # A completed action is not an open slot, and the block says `done`
        # rather than looking like a request. That distinction is the entire
        # subject of data/eval/stale-context.jsonl.
        rendered = render_example("Ring him.", self.STATE, SCHEMA)
        self.assertEqual(
            rendered,
            '[history]\ndone tool=make_call args={"contact":"Anna"}\n[/history]\n\nRing him.',
        )

    def test_history_and_pending_both_render_history_first(self):
        state = {
            "pending": {"tool": "set_timer", "known": {}, "missing": ["minutes"]},
            "priorTurns": [{"tool": "set_timer", "arguments": {"minutes": 5}}],
        }
        rendered = render_state(state, SCHEMA)
        self.assertLess(rendered.index("[history]"), rendered.index("[pending]"))

    def test_argument_order_follows_the_schema(self):
        state = {
            "pending": None,
            "priorTurns": [{"tool": "send_sms", "arguments": {"text": "hi", "contact": "Anna"}}],
        }
        self.assertIn('args={"contact":"Anna","text":"hi"}', render_state(state, SCHEMA))


class DeterminismTests(unittest.TestCase):
    def test_every_case_in_the_repository_renders_identically_on_repeat(self):
        cases = load_corpus(REPO / "data" / "train")[0] + load_corpus(REPO / "data" / "eval")[0]
        self.assertGreater(len(cases), 100)
        for case in cases:
            first = render_example(case.utterance, case.state, SCHEMA)
            second = render_example(case.utterance, case.state, SCHEMA)
            self.assertEqual(first, second, case.id)

    def test_a_stateless_case_never_gains_a_block(self):
        cases = load_corpus(REPO / "data" / "eval")[0]
        stateless = [c for c in cases if c.state is None]
        self.assertGreater(len(stateless), 10)
        for case in stateless:
            self.assertEqual(render_example(case.utterance, case.state, SCHEMA), case.utterance, case.id)


if __name__ == "__main__":
    unittest.main()
