"""Render the reasoning-aligned diagnostic cases into ``needle finetune`` JSONL.

DIAGNOSTIC ONLY - not the training corpus. It differs from
``vesta_router.training`` in exactly the two ways the serving path requires:

``tools``      passed as a LIST, so the trainer serialises it with
               ``separators=(",", ":")`` - the bytes the native engine builds its
               static prefix from (715 prefix tokens for schema v2 whatever
               whitespace the caller's tools JSON has). ``vesta_router.training``
               passes a pre-spaced STRING, which the trainer embeds verbatim:
               a 1066-token tools block the engine never shows the model.
``reasoning``  every example carries one short line, which the trainer wraps in
               ``<think>\\n...\\n</think>\\n`` ahead of the call - the phase the
               engine forces on every turn.

Nothing is written with ``sort_keys``. The engine keeps the tools' key order
(716 prefix tokens in insertion order, 712 sorted; the engine reports 715 with
the same -1 offset as every other configuration measured), and the base model
emits ``{"name":...,"arguments":...}``. ``vesta_router.training`` sorts, so r1
and r2 learned ``{"arguments":...,"name":...}`` - a third difference.

``query``, ``system`` and ``answers`` come from the same places the evaluator and
``vesta_router.training`` take them from.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vesta_router.corpus import load_corpus, validate_corpus  # noqa: E402
from vesta_router.engines import tool_definitions  # noqa: E402
from vesta_router.engines.needle import NEEDLE_SYSTEM_PROMPT  # noqa: E402
from vesta_router.render import render_example  # noqa: E402
from vesta_router.schema import load_tool_schema  # noqa: E402

CASES = Path(__file__).with_name("cases")
SCHEMA = ROOT / "tools/tool-schema-v2.json"


def main(out: str) -> None:
    schema = load_tool_schema(SCHEMA)
    cases, problems = load_corpus(CASES)
    problems += validate_corpus(cases, schema)
    missing = [c.id for c in cases if not str(c.raw.get("reasoning") or "").strip()]
    if problems or missing:
        raise SystemExit(f"invalid diagnostic cases: {[str(p) for p in problems]} no reasoning: {missing}")
    tools = tool_definitions(SCHEMA)
    lines = []
    for case in cases:
        outcome = case.outcome
        answers = (
            [{"name": outcome["tool"], "arguments": outcome["arguments"]}]
            if outcome["kind"] == "tool"
            else []
        )
        example = {
            "id": case.id,
            "outcome": outcome["kind"],
            "system": NEEDLE_SYSTEM_PROMPT,
            "tools": tools,
            "query": render_example(case.utterance, case.state, schema),
            "reasoning": case.raw["reasoning"],
            "answers": answers,
        }
        lines.append(json.dumps(example, ensure_ascii=False) + "\n")
    data = "".join(lines).encode("utf-8")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_bytes(data)
    print(json.dumps({"path": out, "examples": len(lines), "sha256": hashlib.sha256(data).hexdigest()}))


if __name__ == "__main__":
    main(sys.argv[1])
