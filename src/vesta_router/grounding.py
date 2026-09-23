"""The evaluator-side grounding screen.

WHAT THIS IS

A neutral, small re-statement of the rule Vesta enforces at dispatch:

    A tool call may be admitted only when every REQUIRED argument has evidence
    in the user's own words.

It exists so a baseline report can answer two questions instead of one — what
the engine produced, and what would have survived a deterministic gate —
because a model that leans on the guard every time will be wrong the day the
guard has a gap. Hiding fabrication behind a screen that catches it is the
specific failure `data/eval/missing-slot.jsonl` exists to measure.

WHAT THIS IS NOT

It is **not** Vesta's guard, and every report says so. Vesta's lives in
`apps/mobile/lib/scheduling/grounding.ts`, and the spike's generalization of it
in `apps/mobile/lib/router-spike/grounding.ts`; both are TypeScript, both are
wired to the app's own parsers, and copying either here would mean maintaining
a third implementation of a rule that already has two. What is mirrored is the
*shape*: per-slot evidence rules, fail-closed for an argument with no rule, and
deliberately no parameter for confidence — a fabricated ``07:00`` arrived at
1.00.

It is also **not a correctness check**. ``"Set a timer for two hours."``
answered with ``minutes: 2`` is *grounded* (the user did say a duration) and
wrong by a factor of 60. The guard admits it; the corpus catches it.
Normalizing here would turn a measurable defect into a hidden one.

WHAT COUNTS AS "THE USER'S OWN WORDS"

The current utterance, plus the values Vesta itself carried in
``state.pending.known`` — a slot the app supplied and the router echoed back
was not invented by the router.

``state.priorTurns`` is deliberately **excluded**, even though it is rendered
into the prompt the model sees. A completed action is not an open slot, and
reaching into one for a value the user did not repeat is the spike's finding 4.
Including the history block here would ground exactly the failure
`data/eval/stale-context.jsonl` was written to catch.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .schema import ToolSchema

__all__ = [
    "ArgumentVerdict",
    "CallVerdict",
    "REQUIRED_EVIDENCE",
    "has_evidence_rule",
    "screen_call",
]

# ── text normalization ──────────────────────────────────────────────────────

_APOSTROPHES = dict.fromkeys(map(ord, "'’ʼ"), None)
_SPACE = re.compile(r"\s+")
_NON_WORD = re.compile(r"[^\w\s]", re.UNICODE)


def _fold(text: object) -> str:
    """Casefolded, accent-folded, apostrophe-stripped, whitespace-collapsed.

    Accent folding is what lets an Italian evidence rule match text a
    speech-to-text pass wrote without accents, which it routinely does.
    """
    folded = unicodedata.normalize("NFKD", str(text))
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return _SPACE.sub(" ", folded.translate(_APOSTROPHES).casefold()).strip()


# Tokens that carry no evidence on their own, so requiring them to appear would
# make the overlap rule refuse a perfectly grounded paraphrase.
_STOPWORDS = frozenset(
    """a an and at by for from in into me my of on or the to with
    il lo la i gli le un uno una di da con su per tra fra mi che""".split()
)


def _tokens(text: object) -> list[str]:
    return [t for t in _SPACE.split(_NON_WORD.sub(" ", _fold(text))) if t]


# ── evidence kinds ──────────────────────────────────────────────────────────
#
# Each returns the text that grounded the value, or None. A string rather than
# a boolean for the reason the spike gives: a guard that refuses has to be able
# to say what it looked for, and a guard that admits has to show what convinced
# it.

_NUMBER_WORDS = (
    r"a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"fifteen|twenty|thirty|forty|forty-five|fifty|sixty|ninety|"
    r"un|uno|una|due|tre|quattro|cinque|sei|sette|otto|nove|dieci|undici|"
    r"dodici|quindici|venti|trenta|quaranta|cinquanta|sessanta|novanta"
)

_DURATION_UNITS = (
    r"s|sec|secs|second|seconds|m|min|mins|minute|minutes|h|hr|hrs|hour|hours|"
    r"day|days|week|weeks|"
    r"secondo|secondi|minuto|minuti|ora|ore|giorno|giorni|settimana|settimane"
)

_DURATION_PATTERNS = [
    re.compile(rf"\b\d+(?:[.,]\d+)?\s*(?:{_DURATION_UNITS})\b"),
    re.compile(rf"\b(?:{_NUMBER_WORDS})\s+(?:{_DURATION_UNITS})\b"),
    re.compile(r"\b(?:half|quarter)\s+(?:an?\s+)?(?:hour|minute)\b"),
    re.compile(r"\b(?:mezz|mezza|mezzo)\s*ora\b"),
]

_CLOCK_PATTERNS = [
    re.compile(r"\b\d{1,2}\s*[:.]\s*\d{2}\b"),
    re.compile(r"\b\d{1,2}\s*(?:a\.?m\.?|p\.?m\.?)\b"),
    re.compile(r"\bo\s?clock\b"),
    re.compile(r"\b(?:noon|midday|midnight|mezzogiorno|mezzanotte)\b"),
    re.compile(r"\b(?:at|by|before|after|around|alle|all|ore|verso)\s+\d{1,2}\b"),
    # Spelled-out clock times, which the assist surface produces constantly:
    # "seven thirty", "half six", "sette e mezza".
    re.compile(
        rf"\b(?:{_NUMBER_WORDS})\s+(?:o\s?clock|thirty|fifteen|forty-five|"
        rf"e\s+mezza|e\s+un\s+quarto)\b"
    ),
    re.compile(rf"\bhalf\s+(?:past\s+)?(?:{_NUMBER_WORDS})\b"),
]

_DATE_PATTERNS = [
    re.compile(
        r"\b(?:today|tonight|tomorrow|monday|tuesday|wednesday|thursday|friday|"
        r"saturday|sunday|oggi|stasera|domani|lunedi|martedi|mercoledi|giovedi|"
        r"venerdi|sabato|domenica)\b"
    ),
    re.compile(
        r"\b(?:january|february|march|april|may|june|july|august|september|"
        r"october|november|december|gennaio|febbraio|marzo|aprile|maggio|"
        r"giugno|luglio|agosto|settembre|ottobre|novembre|dicembre)\b"
    ),
    re.compile(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b"),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
]


def _first_match(haystack: str, patterns: list[re.Pattern[str]]) -> str | None:
    for pattern in patterns:
        hit = pattern.search(haystack)
        if hit:
            return hit.group(0).strip()
    return None


def _duration_evidence(haystack: str, value: object) -> str | None:
    """A duration the user stated.

    The VALUE never appears verbatim — "two hours" is 120 — so what has to be
    present is a duration, not the number.
    """
    del value
    return _first_match(haystack, _DURATION_PATTERNS)


def _clock_evidence(haystack: str, value: object) -> str | None:
    """A clock time the user stated.

    A duration counts: "wake me in twenty minutes" names a moment as surely as
    "wake me at 7:30" does, and refusing it would be the false negative this
    check is least entitled to.
    """
    del value
    return _first_match(haystack, _CLOCK_PATTERNS) or _first_match(haystack, _DURATION_PATTERNS)


def _when_evidence(haystack: str, value: object) -> str | None:
    """A time OR a day.

    ``create_event.start`` and ``set_reminder.datetime`` are satisfied by
    either, and requiring both would refuse "Friday at noon" for lacking a year.
    """
    return _clock_evidence(haystack, value) or _first_match(haystack, _DATE_PATTERNS)


def _overlap_evidence(haystack: str, value: object) -> str | None:
    """Every content token of the value appears in the input.

    The right test for a slot the model should be COPYING or trimming rather
    than deriving: a destination, a contact, a message body, a search query.
    Token-wise rather than substring-wise because a query is a legitimate
    reduction of a sentence — "insurance policy excess" out of "What does my
    insurance policy say about excess?" — while an invented one is not.

    A token also counts as present when the input holds it with a short suffix
    attached — "Marco" against "What's Marco's number?", which folds to
    "marcos". Refusing a contact search for a possessive would be a false
    refusal on one of the most ordinary utterances there is.

    Fails closed on an empty token set: a value made entirely of stopwords has
    nothing to ground.
    """
    wanted = [t for t in _tokens(value) if t not in _STOPWORDS]
    if not wanted:
        return None
    present = set(_tokens(haystack))
    return " ".join(wanted) if all(_token_present(t, present) for t in wanted) else None


def _token_present(token: str, present: set[str]) -> bool:
    if token in present:
        return True
    if len(token) < 3:
        return False
    return any(word.startswith(token) and len(word) - len(token) <= 2 for word in present)


#: The evidence rule for every REQUIRED argument of every tool in schema v2.
#:
#: Exhaustive by test, not by hope: ``tests/test_grounding.py`` walks the schema
#: and fails if a required argument has no rule. A required argument with no
#: rule is treated as UNGROUNDED at runtime too, so forgetting one produces a
#: refusal rather than a silent admission.
REQUIRED_EVIDENCE = {
    "set_alarm": {"time": _clock_evidence},
    "create_event": {"title": _overlap_evidence, "start": _when_evidence},
    "set_reminder": {"text": _overlap_evidence, "datetime": _when_evidence},
    "set_timer": {"minutes": _duration_evidence},
    "navigate_to": {"destination": _overlap_evidence},
    "search_contacts": {"query": _overlap_evidence},
    "make_call": {"contact": _overlap_evidence},
    "send_sms": {"contact": _overlap_evidence, "text": _overlap_evidence},
    "query_document": {"query": _overlap_evidence},
}


def has_evidence_rule(tool: str, argument: str) -> bool:
    return argument in REQUIRED_EVIDENCE.get(tool, {})


# ── the executability floor ─────────────────────────────────────────────────
#
# Separate from grounding, and it has to be: a value can be perfectly grounded
# and still not dispatchable. Both rules below are the production dispatcher's
# own, read out of the app rather than invented here.


def _is_supplied(tool: str, name: str, value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and not value.strip():
        # `make_call {"contact": ""}` is a real base-model output. It is a call
        # in shape and nothing at all in substance.
        return False
    if tool == "set_timer" and name == "minutes":
        # apps/mobile/lib/native/system-actions.ts refuses `minutes <= 0`
        # outright, and rounds `minutes * 60` to whole seconds — so a duration
        # under half a second is not a timer either.
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return False
        return round(float(value) * 60) >= 1
    return True


@dataclass(frozen=True)
class ArgumentVerdict:
    name: str
    #: ``grounded`` | ``ungrounded`` | ``missing``
    state: str
    #: The text that grounded it, when something did.
    evidence: str | None = None


@dataclass(frozen=True)
class CallVerdict:
    tool: str
    #: Whether a dispatcher would be allowed to run this call.
    admitted: bool
    #: Why not, in one line. ``None`` when admitted.
    reason: str | None
    #: One entry per REQUIRED argument, in the schema's declared order.
    required_arguments: tuple[ArgumentVerdict, ...]

    @property
    def missing(self) -> tuple[str, ...]:
        return tuple(a.name for a in self.required_arguments if a.state == "missing")

    @property
    def ungrounded(self) -> tuple[str, ...]:
        return tuple(a.name for a in self.required_arguments if a.state == "ungrounded")


def _same_value(left: object, right: object) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return float(left) == float(right)
    if isinstance(left, str) and isinstance(right, str):
        return _fold(left) == _fold(right)
    return left == right


def screen_call(
    tool_name: str,
    arguments: dict,
    utterance: str,
    schema: ToolSchema,
    pending_known: dict | None = None,
) -> CallVerdict:
    """Screen one proposed call.

    Note the parameters: the call, the user's words, and the state Vesta chose
    to carry. There is deliberately nowhere to pass a confidence score.
    """
    tool = schema.tool(tool_name)
    if tool is None:
        return CallVerdict(
            tool=tool_name,
            admitted=False,
            reason=f"unknown tool {tool_name!r}",
            required_arguments=(),
        )

    haystack = _fold(utterance)
    carried = pending_known or {}
    rules = REQUIRED_EVIDENCE.get(tool_name, {})
    verdicts: list[ArgumentVerdict] = []

    for name in tool.required:
        value = arguments.get(name)
        if not _is_supplied(tool_name, name, value):
            verdicts.append(ArgumentVerdict(name, "missing"))
            continue
        if name in carried and _same_value(carried[name], value):
            # Vesta supplied this one. Echoing it back is not inventing it.
            verdicts.append(ArgumentVerdict(name, "grounded", "carried from pending state"))
            continue
        rule = rules.get(name)
        if rule is None:
            # Fail closed. An argument nobody wrote a rule for is not thereby safe.
            verdicts.append(ArgumentVerdict(name, "ungrounded"))
            continue
        evidence = rule(haystack, value)
        verdicts.append(ArgumentVerdict(name, "grounded" if evidence else "ungrounded", evidence))

    missing = [v.name for v in verdicts if v.state == "missing"]
    ungrounded = [v.name for v in verdicts if v.state == "ungrounded"]
    if missing:
        reason = f"{tool_name} is missing required {', '.join(missing)}"
    elif ungrounded:
        reason = (
            f"{tool_name} was given {', '.join(ungrounded)}, and the request contains "
            "nothing it could have come from"
        )
    else:
        reason = None

    return CallVerdict(
        tool=tool_name,
        admitted=reason is None,
        reason=reason,
        required_arguments=tuple(verdicts),
    )
