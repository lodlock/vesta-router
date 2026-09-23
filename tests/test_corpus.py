"""Corpus validation, and the corpora themselves.

Two kinds of test here, and they fail for different reasons:

* the validator catches what it claims to catch (constructed bad cases);
* the corpora in this repository are actually valid (the real files).

The second is the one that runs in CI on every pull request.
"""

import unittest
from pathlib import Path

from vesta_router.corpus import (
    Case,
    check_disjoint,
    load_corpus,
    normalize_utterance,
    validate_corpus,
)
from vesta_router.schema import load_tool_schema

REPO = Path(__file__).resolve().parents[1]
SCHEMA = load_tool_schema(REPO / "tools" / "tool-schema-v2.json")
V1_SCHEMA = load_tool_schema(REPO / "tools" / "tool-schema-v1.json")


def case(**overrides) -> Case:
    """A valid case, for tests that break exactly one thing about it."""
    base = dict(
        id="x.1",
        utterance="Set a timer for 20 minutes.",
        locale="en",
        outcome={"kind": "tool", "tool": "set_timer", "arguments": {"minutes": 20}},
        state=None,
        tags=(),
        source="authored",
        category=None,
        file="test.jsonl",
        line=1,
    )
    base.update(overrides)
    return Case(**base)


def reasons(problems) -> str:
    return "\n".join(str(p) for p in problems)


class ValidatorTests(unittest.TestCase):
    def test_a_good_case_passes(self):
        self.assertEqual(validate_corpus([case()], SCHEMA), [])

    def test_unknown_tool(self):
        problems = validate_corpus(
            [case(outcome={"kind": "tool", "tool": "control_light", "arguments": {}})], SCHEMA
        )
        self.assertIn("not in tool schema v2", reasons(problems))

    def test_missing_required_argument_in_a_tool_outcome(self):
        # An outcome that asserts an action the app could not execute.
        problems = validate_corpus(
            [case(outcome={"kind": "tool", "tool": "set_timer", "arguments": {}})], SCHEMA
        )
        self.assertIn("omits required argument `minutes`", reasons(problems))

    def test_argument_that_is_not_on_the_tool(self):
        problems = validate_corpus(
            [case(outcome={"kind": "tool", "tool": "set_timer", "arguments": {"minutes": 20, "seconds": 5}})],
            SCHEMA,
        )
        self.assertIn("has no argument `seconds`", reasons(problems))

    def test_argument_of_the_wrong_type(self):
        problems = validate_corpus(
            [case(outcome={"kind": "tool", "tool": "set_timer", "arguments": {"minutes": "20"}})], SCHEMA
        )
        self.assertIn("must be a number", reasons(problems))

    def test_enum_argument(self):
        problems = validate_corpus(
            [case(outcome={"kind": "tool", "tool": "get_time", "arguments": {"kind": "sideways"}})], SCHEMA
        )
        self.assertIn("must be one of", reasons(problems))

    def test_incomplete_must_name_something_missing(self):
        # An `incomplete` with an empty list asserts nothing and would pass
        # every other check. It is the shape a half-written case takes.
        problems = validate_corpus(
            [case(outcome={"kind": "incomplete", "tool": "set_timer", "missing": []})], SCHEMA
        )
        self.assertIn("at least one missing argument", reasons(problems))

    def test_incomplete_cannot_miss_an_optional_argument(self):
        problems = validate_corpus(
            [case(outcome={"kind": "incomplete", "tool": "set_timer", "missing": ["label"]})], SCHEMA
        )
        self.assertIn("not a required argument", reasons(problems))

    def test_chat_must_not_name_a_tool(self):
        problems = validate_corpus([case(outcome={"kind": "chat", "tool": "set_timer"})], SCHEMA)
        self.assertIn("must not name a tool", reasons(problems))

    def test_unknown_outcome_kind(self):
        problems = validate_corpus([case(outcome={"kind": "escalate"})], SCHEMA)
        self.assertIn("is not one of", reasons(problems))

    def test_unknown_locale(self):
        self.assertIn("locale", reasons(validate_corpus([case(locale="de")], SCHEMA)))

    def test_duplicate_id(self):
        problems = validate_corpus([case(), case(line=2)], SCHEMA)
        self.assertIn("duplicate id", reasons(problems))

    def test_every_problem_is_collected(self):
        problems = validate_corpus(
            [case(locale="de", outcome={"kind": "tool", "tool": "set_timer", "arguments": {}})], SCHEMA
        )
        self.assertGreaterEqual(len(problems), 2)


class StateValidationTests(unittest.TestCase):
    def test_pending_tool_must_exist(self):
        problems = validate_corpus(
            [case(state={"pending": {"tool": "open_app", "known": {}, "missing": []}})], SCHEMA
        )
        self.assertIn("is not in the schema", reasons(problems))

    def test_pending_known_argument_must_exist(self):
        problems = validate_corpus(
            [case(state={"pending": {"tool": "set_timer", "known": {"seconds": 30}, "missing": []}})], SCHEMA
        )
        self.assertIn("is not on set_timer", reasons(problems))

    def test_a_slot_cannot_be_both_known_and_missing(self):
        # The state a hand-edited case lands in after a rename.
        problems = validate_corpus(
            [case(state={"pending": {"tool": "set_timer", "known": {"minutes": 5}, "missing": ["minutes"]}})],
            SCHEMA,
        )
        self.assertIn("both known and missing", reasons(problems))

    def test_prior_turn_tool_must_exist(self):
        problems = validate_corpus(
            [case(state={"pending": None, "priorTurns": [{"tool": "control_light", "arguments": {}}]})],
            SCHEMA,
        )
        self.assertIn("prior turn tool", reasons(problems))


class DisjointnessTests(unittest.TestCase):
    def test_shared_id_is_a_problem(self):
        problems = check_disjoint([case(id="a", file="train.jsonl")], [case(id="a", file="eval.jsonl")])
        self.assertIn("id also appears in training", reasons(problems))

    def test_shared_utterance_is_a_problem_even_with_different_ids(self):
        # Ids are easy to keep distinct and prove nothing. The same sentence
        # under two names is the actual memorization risk.
        problems = check_disjoint(
            [case(id="tr.a", utterance="Set a timer for 20 minutes.", file="train.jsonl")],
            [case(id="ev.a", utterance="set a timer for 20 minutes!", file="eval.jsonl")],
        )
        self.assertIn("measures memorization", reasons(problems))

    def test_normalization_folds_case_accents_and_punctuation(self):
        self.assertEqual(normalize_utterance("Metti un timer, per favore!"), "metti un timer per favore")
        self.assertEqual(normalize_utterance("È ORA?"), "e ora")


class RealCorpusTests(unittest.TestCase):
    """The files in this repository, checked as CI checks them."""

    @classmethod
    def setUpClass(cls):
        cls.train, cls.train_load = load_corpus(REPO / "data" / "train")
        cls.evaluation, cls.eval_load = load_corpus(REPO / "data" / "eval")

    def test_both_corpora_are_non_empty(self):
        # Without this every assertion below would pass vacuously if a directory
        # moved or a glob stopped matching.
        self.assertGreater(len(self.train), 50)
        self.assertGreater(len(self.evaluation), 40)

    def test_both_corpora_parse(self):
        self.assertEqual(reasons(self.train_load), "")
        self.assertEqual(reasons(self.eval_load), "")

    def test_both_corpora_validate_against_schema_v2(self):
        self.assertEqual(reasons(validate_corpus(self.train, SCHEMA)), "")
        self.assertEqual(reasons(validate_corpus(self.evaluation, SCHEMA)), "")

    def test_corpora_are_disjoint(self):
        self.assertEqual(reasons(check_disjoint(self.train, self.evaluation)), "")

    def test_both_languages_are_present_in_both_corpora(self):
        # Italian is a day-one language for Vesta, so it is a day-one language
        # for the router. A corpus that drifts to English-only produces a router
        # that is English-only, and nothing else would notice.
        for name, cases in [("train", self.train), ("eval", self.evaluation)]:
            locales = {c.locale for c in cases}
            self.assertEqual(locales, {"en", "it"}, f"{name} corpus locales")

    def test_the_device_observed_failures_are_still_cases(self):
        # The spike's findings 2 and 4. If one disappears from the corpus, the
        # regression it records stops being checked by anything.
        by_id = {c.id: c for c in self.evaluation}

        self.assertEqual(
            by_id["slot.alarm.bare"].outcome,
            {"kind": "incomplete", "tool": "set_alarm", "missing": ["time"]},
        )
        self.assertEqual(by_id["stale.timer.prior-turn"].outcome["kind"], "incomplete")
        for case_id in [
            "slot.timer.bare",
            "slot.alarm.bare",
            "stale.timer.prior-turn",
            "stale.alarm.prior-turn",
        ]:
            self.assertEqual(by_id[case_id].source, "spike-device-2026-09-22", case_id)

    def test_the_paired_utterance_with_and_without_state(self):
        # slot.followup.minutes and neg.bare-duration are both "20 minutes" and
        # must classify differently. Nothing but the state block can tell them
        # apart, which is the whole argument for supplying state explicitly.
        by_id = {c.id: c for c in self.evaluation}
        with_state = by_id["slot.followup.minutes"]
        without = by_id["neg.bare-duration"]

        self.assertEqual(normalize_utterance(with_state.utterance), normalize_utterance(without.utterance))
        self.assertEqual(with_state.state["pending"]["tool"], "set_timer")
        self.assertIsNone(without.state)
        self.assertEqual(with_state.outcome["kind"], "tool")
        self.assertEqual(without.outcome["kind"], "chat")

    def test_high_cost_tools_have_negative_cases(self):
        # make_call and send_sms are irreversible in the way that matters. A
        # corpus with positives and no traps for them teaches eagerness.
        high_cost = [c for c in self.evaluation if "high-cost" in c.tags]
        self.assertGreaterEqual(len(high_cost), 5)
        self.assertTrue(any(c.outcome["kind"] == "chat" for c in high_cost))
        self.assertTrue(any(c.outcome["kind"] == "incomplete" for c in high_cost))

    def test_every_tool_in_the_schema_appears_somewhere(self):
        # A tool with no cases is a tool nothing measures.
        covered = {c.outcome.get("tool") for c in self.train + self.evaluation if c.outcome.get("tool")}
        self.assertEqual(set(SCHEMA.tool_names) - covered, set())


class HistoricalCorpusTests(unittest.TestCase):
    """data/historical is frozen v1 material and is validated against v1."""

    def test_it_still_parses_and_validates_against_the_v1_schema(self):
        cases, load_problems = load_corpus(REPO / "data" / "historical")
        self.assertGreater(len(cases), 50)
        self.assertEqual(reasons(load_problems), "")
        self.assertEqual(reasons(validate_corpus(cases, V1_SCHEMA)), "")

    def test_it_would_NOT_validate_against_v2(self):
        # The point of freezing it. v1's set_timer took duration_seconds and its
        # control_light and open_app do not exist in Vesta's registry, so these
        # cases are readable history and not a training target.
        cases, _ = load_corpus(REPO / "data" / "historical")
        self.assertNotEqual(validate_corpus(cases, SCHEMA), [])

    def test_it_is_not_mixed_into_the_live_corpora(self):
        live_ids = {c.id for c in load_corpus(REPO / "data" / "train")[0]}
        live_ids |= {c.id for c in load_corpus(REPO / "data" / "eval")[0]}
        historical_ids = {c.id for c in load_corpus(REPO / "data" / "historical")[0]}
        self.assertEqual(live_ids & historical_ids, set())


if __name__ == "__main__":
    unittest.main()
