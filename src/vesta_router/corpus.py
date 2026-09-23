"""Loading and validating the training and evaluation corpora.

Every problem is collected, never raised on the first. A contributor fixing a
corpus wants the whole list in one pass, and CI wants to print it.

The rules below are the ones that would otherwise be caught by a human reading a
diff, or not caught at all:

* a tool that is not in the schema — usually a rename that missed a file;
* a required argument absent from a ``tool`` outcome, which asserts an
  unexecutable action;
* a ``missing`` list naming something that is not a required argument, which
  asserts a refusal for a reason that does not exist;
* an ``incomplete`` outcome whose ``missing`` list is empty, which asserts
  nothing at all;
* a duplicate id, which makes a cited result ambiguous;
* an eval utterance that also appears in training, which measures memorization.

The last one is checked on a normalized form of the utterance, not on the id.
Ids are easy to keep distinct and prove nothing; the same sentence under two
names is the actual failure.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from .schema import OUTCOME_KINDS, ToolSchema

__all__ = ["Case", "Problem", "load_corpus", "validate_corpus", "check_disjoint", "normalize_utterance"]

LOCALES = ("en", "it")


@dataclass(frozen=True)
class Problem:
    file: str
    line: int
    case_id: str
    reason: str

    def __str__(self) -> str:
        where = f"{self.file}:{self.line}"
        return f"{where} [{self.case_id or '?'}] {self.reason}"


@dataclass
class Case:
    id: str
    utterance: str
    locale: str
    outcome: dict
    state: dict | None
    tags: tuple[str, ...]
    source: str
    category: str | None
    file: str
    line: int
    raw: dict = field(repr=False, default_factory=dict)


def _outcome_of(doc: dict) -> dict | None:
    """Eval cases carry ``expect``; training cases carry ``output``.

    Two names for one shape, kept apart on purpose: an eval case states what
    must happen and a training case states what to learn, and a file that
    confuses them is a file in the wrong directory.
    """
    if "expect" in doc:
        return doc["expect"] if isinstance(doc["expect"], dict) else None
    if "output" in doc:
        return doc["output"] if isinstance(doc["output"], dict) else None
    return None


def load_corpus(directory: str | Path) -> tuple[list[Case], list[Problem]]:
    """Read every ``*.jsonl`` in a directory. Malformed lines become problems."""
    directory = Path(directory)
    cases: list[Case] = []
    problems: list[Problem] = []

    for path in sorted(directory.glob("*.jsonl")):
        name = path.name
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError as e:
                problems.append(Problem(name, index, "", f"not valid JSON: {e.msg}"))
                continue
            if not isinstance(doc, dict):
                problems.append(Problem(name, index, "", "a case must be a JSON object"))
                continue

            outcome = _outcome_of(doc)
            if outcome is None:
                problems.append(
                    Problem(
                        name, index, str(doc.get("id", "")), "has neither an `expect` nor an `output` object"
                    )
                )
                continue

            cases.append(
                Case(
                    id=doc.get("id", ""),
                    utterance=doc.get("utterance", ""),
                    locale=doc.get("locale", ""),
                    outcome=outcome,
                    state=doc.get("state"),
                    tags=tuple(doc.get("tags", ())),
                    source=doc.get("source", ""),
                    category=doc.get("category"),
                    file=name,
                    line=index,
                    raw=doc,
                )
            )

    return cases, problems


def _failer(problems: list[Problem], case: Case):
    """A recorder bound to one case.

    A closure defined inside the loop would do the same thing and would be safe,
    because it is only ever called within its own iteration - but it is the
    shape that stops being safe the moment somebody defers a call, so it is
    bound explicitly instead.
    """

    def fail(reason: str) -> None:
        problems.append(Problem(case.file, case.line, case.id, reason))

    return fail


def validate_corpus(cases: list[Case], schema: ToolSchema) -> list[Problem]:
    problems: list[Problem] = []
    seen_ids: dict[str, Problem] = {}

    for case in cases:
        fail = _failer(problems, case)

        if not isinstance(case.id, str) or not case.id:
            fail("has no id")
        elif case.id in seen_ids:
            first = seen_ids[case.id]
            fail(f"duplicate id, first seen at {first.file}:{first.line}")
        else:
            seen_ids[case.id] = Problem(case.file, case.line, case.id, "")

        if not isinstance(case.utterance, str) or not case.utterance.strip():
            fail("has no utterance")
        if case.locale not in LOCALES:
            fail(f"locale {case.locale!r} is not one of {', '.join(LOCALES)}")

        kind = case.outcome.get("kind")
        if kind not in OUTCOME_KINDS:
            fail(f"outcome kind {kind!r} is not one of {', '.join(OUTCOME_KINDS)}")
            continue

        if kind == "chat":
            if case.outcome.get("tool"):
                fail("a chat outcome must not name a tool")
            continue

        tool_name = case.outcome.get("tool")
        tool = schema.tool(tool_name) if isinstance(tool_name, str) else None
        if tool is None:
            fail(f"tool {tool_name!r} is not in tool schema v{schema.version}")
            continue

        if kind == "cancel":
            continue

        if kind == "tool":
            arguments = case.outcome.get("arguments")
            if not isinstance(arguments, dict):
                fail("a tool outcome needs an `arguments` object")
                continue
            for name in tool.required:
                if name not in arguments:
                    fail(f"{tool.name} outcome omits required argument `{name}`")
            for name, value in arguments.items():
                spec = tool.argument(name)
                if spec is None:
                    fail(f"{tool.name} has no argument `{name}`")
                    continue
                if spec.type == "number" and not isinstance(value, (int, float)):
                    fail(f"{tool.name}.{name} must be a number, got {type(value).__name__}")
                if spec.type == "string" and not isinstance(value, str):
                    fail(f"{tool.name}.{name} must be a string, got {type(value).__name__}")
                if spec.enum and value not in spec.enum:
                    fail(f"{tool.name}.{name} must be one of {', '.join(spec.enum)}")

        elif kind == "incomplete":
            missing = case.outcome.get("missing")
            if not isinstance(missing, list) or not missing:
                # An `incomplete` with nothing missing asserts nothing. It is
                # the shape a half-written case takes, and it would pass every
                # other check here.
                fail("an incomplete outcome must list at least one missing argument")
                continue
            for name in missing:
                if name not in tool.required:
                    fail(f"`{name}` is not a required argument of {tool.name}, so it cannot be missing")

        if case.state is not None:
            problems.extend(_validate_state(case, schema))

    return problems


def _validate_state(case: Case, schema: ToolSchema) -> list[Problem]:
    problems: list[Problem] = []

    def fail(reason: str) -> None:
        problems.append(Problem(case.file, case.line, case.id, reason))

    state = case.state
    if not isinstance(state, dict):
        fail("`state` must be an object or null")
        return problems

    pending = state.get("pending")
    if pending is not None:
        if not isinstance(pending, dict):
            fail("`state.pending` must be an object or null")
        else:
            tool = schema.tool(pending.get("tool"))
            if tool is None:
                fail(f"pending tool {pending.get('tool')!r} is not in the schema")
            else:
                known = pending.get("known", {})
                if not isinstance(known, dict):
                    fail("`state.pending.known` must be an object")
                else:
                    for name in known:
                        if tool.argument(name) is None:
                            fail(f"pending known argument `{name}` is not on {tool.name}")
                missing = pending.get("missing", [])
                if not isinstance(missing, list):
                    fail("`state.pending.missing` must be a list")
                else:
                    for name in missing:
                        if name not in tool.required:
                            fail(f"pending missing `{name}` is not required by {tool.name}")
                    overlap = set(missing) & set(known if isinstance(known, dict) else {})
                    if overlap:
                        # A slot cannot be both supplied and absent. This is the
                        # state a hand-edited case lands in after a rename.
                        fail(f"pending lists {', '.join(sorted(overlap))} as both known and missing")

    for turn in state.get("priorTurns", []) or []:
        if not isinstance(turn, dict):
            fail("`state.priorTurns` entries must be objects")
            continue
        if schema.tool(turn.get("tool")) is None:
            fail(f"prior turn tool {turn.get('tool')!r} is not in the schema")

    return problems


_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_SPACE = re.compile(r"\s+")


def normalize_utterance(text: str) -> str:
    """Casefolded, accent-folded, punctuation-stripped, whitespace-collapsed.

    Used only for disjointness. It is deliberately aggressive: two cases that
    differ by a comma are the same sentence for the purpose of asking whether
    the model was trained on the thing it is being measured on.
    """
    folded = unicodedata.normalize("NFKD", text)
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return _SPACE.sub(" ", _PUNCT.sub(" ", folded.casefold())).strip()


def check_disjoint(train: list[Case], evaluation: list[Case]) -> list[Problem]:
    """Eval must not overlap training, by id or by utterance."""
    problems: list[Problem] = []

    train_ids = {c.id: c for c in train}
    train_utterances: dict[str, Case] = {}
    for c in train:
        train_utterances.setdefault(normalize_utterance(c.utterance), c)

    for case in evaluation:
        if case.id in train_ids:
            other = train_ids[case.id]
            problems.append(
                Problem(
                    case.file, case.line, case.id, f"id also appears in training at {other.file}:{other.line}"
                )
            )
        key = normalize_utterance(case.utterance)
        if key in train_utterances:
            other = train_utterances[key]
            problems.append(
                Problem(
                    case.file,
                    case.line,
                    case.id,
                    f"utterance also appears in training as {other.id} ({other.file}:{other.line}); "
                    "an eval case that has been trained on measures memorization",
                )
            )

    return problems
