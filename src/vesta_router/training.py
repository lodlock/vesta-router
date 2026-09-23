"""Rendering ``data/train`` into the JSONL the Needle trainer reads.

THE ONE RULE THIS MODULE EXISTS TO HOLD

**A training example must be the same bytes the evaluator would send for the
same turn.** The model is trained on a prompt and then measured on a prompt; if
those differ by a space in the tool JSON or a word in the system prompt, the
measurement is of a prompt the model never saw and every number after it is
about something else.

So three things are taken from where the evaluator takes them, never restated:

``query``   :func:`vesta_router.render.render_example` — the same renderer, so
            the ``[pending]`` and ``[history]`` blocks are byte-identical.
``system``  ``vesta_router.engines.needle.NEEDLE_SYSTEM_PROMPT``, passed in.
``tools``   ``vesta_router.engines.tool_definitions``, **pre-serialized with the
            exact call ``NeedleEngine`` uses** (``json.dumps(tools,
            ensure_ascii=False)``) and written as a STRING. ``needle``'s
            trainer embeds a string field verbatim and re-serializes a list, so
            passing the string is what keeps the two paths identical.

WHAT THE MODEL IS TAUGHT TO SAY

Needle's output vocabulary is a list of calls. It has exactly two shapes, and
three of this project's four outcomes map onto the empty one:

=============  ===========================================================
``tool``       one call, the expected tool with the expected arguments
``incomplete`` **no call** — a required argument has no evidence, so there
               is nothing executable to propose and the app asks instead
``cancel``     **no call** — the user abandoned the action
``chat``       **no call** — the sentence was not a request
=============  ===========================================================

``incomplete`` collapsing to "emit nothing" is a deliberate narrowing and the
corpus README says why it is legitimate: ``incomplete`` is an outcome
assertion — *no executable action may reach dispatch* — not an output
assertion. Withholding satisfies it, and withholding is the only one of the
satisfying behaviours the model can be trained toward, because the engine has
no token for "I would call this tool but I am missing an argument". Vesta still
decides what to ask for; it does not need the router to guess.

``cancel`` genuinely has no representation, and this is the same empty answer a
``chat`` turn gets. That is a known limit of the surface, not a modelling
choice, and it is why ``cancelNoActionRate`` scores only whether an action was
produced.

NO REASONING FIELD

``needle``'s trainer will wrap a ``reasoning`` string in ``<think>`` and train
the model to produce it. Every example here omits it, uniformly. A corpus where
some examples carry authored reasoning and others do not teaches the model that
reasoning is optional in a way that correlates with whatever else distinguishes
those files — and authored reasoning for 130 cases would be 130 opportunities
to teach a wrong derivation. The corpus has none, so none is invented here.

DETERMINISM

Examples are written in corpus order — files sorted by name, lines in order —
so the same corpus renders to the same bytes and the JSONL's own SHA-256 is a
usable provenance field. :func:`write_training_jsonl` returns that digest.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .corpus import Case
from .render import render_example
from .runner import corpus_digest as _digest_paths
from .schema import ToolSchema

__all__ = ["corpus_digest", "needle_example", "render_training_examples", "write_training_jsonl"]


def corpus_digest(directory: str | Path) -> dict:
    """The content digest of every ``*.jsonl`` in a corpus directory."""
    return _digest_paths(sorted(Path(directory).glob("*.jsonl")))


def needle_example(case: Case, schema: ToolSchema, tools_json: str, system_prompt: str) -> dict:
    """One corpus case as one line of the trainer's JSONL.

    ``id`` and ``category`` ride along unread by the trainer. They are what
    makes a rendered corpus traceable back to the file it came from, which is
    the difference between "132 examples" and an auditable training set.
    """
    outcome = case.outcome
    kind = outcome.get("kind")
    if kind == "tool":
        answers = [{"name": outcome.get("tool"), "arguments": outcome.get("arguments") or {}}]
    else:
        answers = []

    return {
        "id": case.id,
        "category": case.category,
        "locale": case.locale,
        "outcome": kind,
        "system": system_prompt,
        "tools": tools_json,
        "query": render_example(case.utterance, case.state, schema),
        "answers": answers,
    }


def render_training_examples(
    cases: list[Case], schema: ToolSchema, tools: list[dict], system_prompt: str
) -> list[dict]:
    tools_json = json.dumps(tools, ensure_ascii=False)
    return [needle_example(case, schema, tools_json, system_prompt) for case in cases]


def write_training_jsonl(
    path: str | Path,
    cases: list[Case],
    schema: ToolSchema,
    tools: list[dict],
    system_prompt: str,
) -> dict:
    """Write the JSONL and describe what was written.

    The digest is of the file as the trainer will read it, so a training run
    can record the exact bytes it consumed rather than the corpus revision it
    was supposedly derived from.
    """
    examples = render_training_examples(cases, schema, tools, system_prompt)
    text = "".join(json.dumps(example, ensure_ascii=False, sort_keys=True) + "\n" for example in examples)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode("utf-8")
    destination.write_bytes(data)

    by_outcome: dict[str, int] = {}
    for example in examples:
        by_outcome[example["outcome"]] = by_outcome.get(example["outcome"], 0) + 1

    return {
        "path": str(destination),
        "examples": len(examples),
        "byOutcome": by_outcome,
        "sha256": hashlib.sha256(data).hexdigest(),
        "sizeBytes": len(data),
    }
