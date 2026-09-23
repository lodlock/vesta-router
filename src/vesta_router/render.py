"""Rendering conversation state into the text the model sees.

WHY STATE IS RENDERED HERE AND NOT STORED PRE-RENDERED

The corpora store ``state`` as a structured object. The obvious alternative —
writing the ``[pending]`` block straight into the utterance field — was rejected
for two reasons that both bite later:

* a corpus of pre-rendered strings cannot be re-rendered when the block format
  changes, and the format WILL change;
* a typo inside a pre-rendered block is invisible to every checker. Structured
  state is validated against the tool schema (``corpus.validate_corpus``);
  a string is not.

WHY EXPLICIT STATE AT ALL

Needle trains best on single-turn examples, and Vesta owns conversation and
pending-action state regardless. So the router is never handed implicit chat
history: it gets exactly what Vesta decided it may use, in a block, on one turn.
The spike's finding 4 is what happens otherwise — the engine reached back into
earlier turns to fill a slot the current prompt left empty, and said so in its
own reasoning text.

The pair ``slot.followup.minutes`` and ``neg.bare-duration`` is the test of it.
Same utterance, "20 minutes". With a pending timer it is a duration; without one
it is not a request. Nothing but this block can tell them apart.

DETERMINISM

The same state must always render to the same bytes, or the model sees two
prompts for one situation and the KV prefix argument Vesta cares about
elsewhere stops holding here too. So:

* ``state is None`` renders NOTHING — not an empty block. A stateless turn must
  produce the bytes a stateless turn actually produces in the app.
* ``known`` is JSON with sorted keys and no spaces after separators.
* ``missing`` is listed in the TOOL SCHEMA's declared argument order, never the
  order somebody typed it into the file.
* ``priorTurns`` renders as ``[history]`` — completed actions, which are not
  open slots, and the distinction is the entire subject of
  ``data/eval/stale-context.jsonl``.
"""

from __future__ import annotations

import json

from .schema import ToolSchema

__all__ = ["render_state", "render_example"]


def _render_known(known: dict) -> str:
    return json.dumps(known, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def render_state(state: dict | None, schema: ToolSchema) -> str:
    """Render a state object to the block that precedes the utterance.

    Returns the empty string when there is no state to show — including for a
    state object that carries neither a pending action nor any prior turns,
    because an empty block and no block are different bytes and only one of them
    is what a stateless turn produces.
    """
    if not state:
        return ""

    blocks: list[str] = []

    prior = state.get("priorTurns") or []
    if prior:
        lines = ["[history]"]
        for turn in prior:
            tool_name = turn.get("tool")
            tool = schema.tool(tool_name)
            arguments = turn.get("arguments") or {}
            if tool is not None:
                # Schema order, so two files listing the same arguments
                # differently still render identically.
                ordered = {a.name: arguments[a.name] for a in tool.arguments if a.name in arguments}
            else:
                ordered = dict(sorted(arguments.items()))
            lines.append(f"done tool={tool_name} args={_render_known(ordered)}")
        lines.append("[/history]")
        blocks.append("\n".join(lines))

    pending = state.get("pending")
    if pending:
        tool_name = pending.get("tool")
        tool = schema.tool(tool_name)
        known = pending.get("known") or {}
        missing = list(pending.get("missing") or [])
        if tool is not None:
            order = {name: i for i, name in enumerate(a.name for a in tool.arguments)}
            missing.sort(key=lambda name: order.get(name, len(order)))
        blocks.append(
            "\n".join(
                [
                    "[pending]",
                    f"tool={tool_name}",
                    f"known={_render_known(known)}",
                    f"missing={','.join(missing)}",
                    "[/pending]",
                ]
            )
        )

    return "\n\n".join(blocks)


def render_example(utterance: str, state: dict | None, schema: ToolSchema) -> str:
    """The full model input for one turn: state block, blank line, utterance."""
    block = render_state(state, schema)
    if not block:
        return utterance
    return f"{block}\n\n{utterance}"
