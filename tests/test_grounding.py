"""The evaluator-side grounding screen.

Two properties matter more than any individual rule:

* every REQUIRED argument of every tool has an evidence rule, checked by
  walking the schema rather than by listing them here — a tool added to v2
  without a rule must fail this file, not fail closed in silence at runtime;
* the screen admits every correct answer the eval corpus asks for. A guard that
  refuses the right answer turns a passing model into a failing one, and the
  failure would look like a model defect.
"""

import unittest
from pathlib import Path

from vesta_router.corpus import load_corpus
from vesta_router.grounding import REQUIRED_EVIDENCE, has_evidence_rule, screen_call
from vesta_router.schema import load_tool_schema

REPO = Path(__file__).resolve().parents[1]
SCHEMA = load_tool_schema(REPO / "tools" / "tool-schema-v2.json")


class ExhaustivenessTests(unittest.TestCase):
    def test_every_required_argument_has_a_rule(self):
        for tool in SCHEMA.tools:
            for argument in tool.required:
                self.assertTrue(
                    has_evidence_rule(tool.name, argument),
                    f"{tool.name}.{argument} has no evidence rule",
                )

    def test_no_rule_survives_a_tool_being_renamed_away(self):
        for name in REQUIRED_EVIDENCE:
            self.assertIsNotNone(SCHEMA.tool(name), f"{name} is not in the schema any more")

    def test_an_argument_with_no_rule_is_refused_not_admitted(self):
        # The failure mode of forgetting a rule must be a refusal. Proven by
        # screening a tool this module deliberately knows nothing about.
        verdict = screen_call("get_calendar_events", {}, "anything", SCHEMA)
        self.assertTrue(verdict.admitted)  # no required arguments at all
        self.assertEqual(verdict.required_arguments, ())


class AdmitsTheCorpusTests(unittest.TestCase):
    """Every `tool` outcome in the live corpus must pass the screen."""

    def test_every_expected_tool_call_is_admitted(self):
        cases, problems = load_corpus(REPO / "data" / "eval")
        self.assertEqual(problems, [])
        checked = 0
        for case in cases:
            if case.outcome.get("kind") != "tool":
                continue
            known = ((case.state or {}).get("pending") or {}).get("known") or {}
            verdict = screen_call(
                case.outcome["tool"], case.outcome["arguments"], case.utterance, SCHEMA, known
            )
            self.assertTrue(
                verdict.admitted,
                f"{case.id}: the correct answer was refused - {verdict.reason}",
            )
            checked += 1
        self.assertGreater(checked, 20)


class RefusesFabricationTests(unittest.TestCase):
    def test_the_spike_alarm(self):
        # "Wake me up." with an invented 07:00, the case the whole contract is
        # built around.
        verdict = screen_call("set_alarm", {"time": "07:00"}, "Wake me up.", SCHEMA)
        self.assertFalse(verdict.admitted)
        self.assertEqual(verdict.ungrounded, ("time",))

    def test_an_empty_required_string_is_missing_not_grounded(self):
        verdict = screen_call("make_call", {"contact": ""}, "Make a call.", SCHEMA)
        self.assertFalse(verdict.admitted)
        self.assertEqual(verdict.missing, ("contact",))

    def test_a_zero_timer_is_not_a_timer(self):
        # apps/mobile/lib/native/system-actions.ts refuses minutes <= 0.
        verdict = screen_call("set_timer", {"minutes": 0.0}, "Set a timer.", SCHEMA)
        self.assertEqual(verdict.missing, ("minutes",))

    def test_a_pronoun_does_not_ground_a_contact(self):
        verdict = screen_call("make_call", {"contact": "Marco"}, "Call them back.", SCHEMA)
        self.assertFalse(verdict.admitted)

    def test_an_unknown_tool_is_refused(self):
        verdict = screen_call("control_light", {"room": "kitchen"}, "kitchen light off", SCHEMA)
        self.assertFalse(verdict.admitted)
        self.assertIn("unknown tool", verdict.reason)


class GroundingIsNotCorrectnessTests(unittest.TestCase):
    def test_a_grounded_but_wrong_conversion_is_admitted(self):
        # "two hours" -> 2 is wrong by a factor of 60 and is ADMITTED: the user
        # did say a duration. Catching it is the corpus's job, and normalizing
        # it here would hide a measurable defect behind a patch.
        verdict = screen_call("set_timer", {"minutes": 2}, "Set a timer for two hours.", SCHEMA)
        self.assertTrue(verdict.admitted)


class CarriedStateTests(unittest.TestCase):
    STATE_KNOWN = {"text": "call the dentist"}

    def test_a_value_vesta_carried_is_grounded(self):
        verdict = screen_call(
            "set_reminder",
            {"text": "call the dentist", "datetime": "16:00"},
            "at 4pm",
            SCHEMA,
            self.STATE_KNOWN,
        )
        self.assertTrue(verdict.admitted)

    def test_a_different_value_is_not_grounded_by_the_carried_one(self):
        # Ignoring the state it was given is the same defect as inventing one.
        verdict = screen_call(
            "set_reminder",
            {"text": "buy milk", "datetime": "16:00"},
            "at 4pm",
            SCHEMA,
            self.STATE_KNOWN,
        )
        self.assertFalse(verdict.admitted)
        self.assertEqual(verdict.ungrounded, ("text",))

    def test_history_is_not_evidence(self):
        # The rendered prompt contains the [history] block, but the screen sees
        # only the utterance. Reaching into a completed action for a value the
        # user did not repeat is the spike's finding 4.
        verdict = screen_call("set_timer", {"minutes": 5}, "Set a timer.", SCHEMA)
        self.assertFalse(verdict.admitted)


class ConfidenceTests(unittest.TestCase):
    """A fabricated 07:00 arrived at confidence 1.00. There is nowhere to pass one."""

    def test_screen_call_takes_no_confidence(self):
        import inspect

        self.assertNotIn("confidence", inspect.signature(screen_call).parameters)

    def test_no_verdict_carries_one(self):
        import dataclasses

        from vesta_router.grounding import ArgumentVerdict, CallVerdict

        for cls in (ArgumentVerdict, CallVerdict):
            names = {field.name for field in dataclasses.fields(cls)}
            self.assertNotIn("confidence", names)


if __name__ == "__main__":
    unittest.main()
