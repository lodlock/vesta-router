"""Loading and reading a tool schema.

A tool schema names the surface a model version was trained on, and it is the
thing every corpus case is validated against. Two exist:

    v1  the Needle spike's four toy tools. FROZEN, historical, never trained on.
    v2  derived from Vesta's production tool registry. The training target.

The difference is not cosmetic. v1's ``set_timer`` took ``duration_seconds``;
v2's takes ``minutes``, because that is what
``apps/mobile/lib/tools/tool-registry.ts`` dispatches. The spike's headline
failure — "Set a timer for 20 minutes." answered with ``duration_seconds: 20``,
a 60x error — is *correct* against v2. That is the clearest possible argument
for deriving the schema from the app rather than inventing one: a whole
benchmark result changed meaning when the argument name did.

There is no JSON-Schema dependency here. The documents are small, the rules are
specific, and the error messages a hand-written reader produces name the tool
and the argument rather than a JSON pointer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Argument", "Tool", "ToolSchema", "load_tool_schema", "OUTCOME_KINDS"]

#: The four things a routing decision can be. `chat` is an OUTCOME, never a
#: tool — see `chatIsNotATool` in tools/tool-schema-v2.json.
OUTCOME_KINDS = ("tool", "incomplete", "cancel", "chat")


@dataclass(frozen=True)
class Argument:
    name: str
    type: str
    required: bool
    enum: tuple[str, ...] | None = None


@dataclass(frozen=True)
class Tool:
    name: str
    arguments: tuple[Argument, ...]

    @property
    def required(self) -> tuple[str, ...]:
        """Required argument names, in the schema's own declared order.

        The order matters: it is what ``render.py`` uses for the ``missing``
        list, so the same state always renders to the same bytes.
        """
        return tuple(a.name for a in self.arguments if a.required)

    @property
    def optional(self) -> tuple[str, ...]:
        return tuple(a.name for a in self.arguments if not a.required)

    def argument(self, name: str) -> Argument | None:
        for a in self.arguments:
            if a.name == name:
                return a
        return None


@dataclass(frozen=True)
class ToolSchema:
    version: int
    status: str
    tools: tuple[Tool, ...]

    def tool(self, name: str) -> Tool | None:
        for t in self.tools:
            if t.name == name:
                return t
        return None

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(t.name for t in self.tools)


def load_tool_schema(path: str | Path) -> ToolSchema:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))

    version = doc.get("toolSchemaVersion")
    if not isinstance(version, int):
        raise ValueError(f"{path}: toolSchemaVersion must be an integer")

    tools: list[Tool] = []
    for entry in doc.get("tools", []):
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{path}: a tool has no name")
        arguments: list[Argument] = []
        for arg_name, spec in (entry.get("parameters") or {}).items():
            if not isinstance(spec, dict):
                raise ValueError(f"{path}: {name}.{arg_name} is not an object")
            arg_type = spec.get("type")
            if arg_type not in ("string", "number", "boolean", "integer"):
                raise ValueError(f"{path}: {name}.{arg_name} has unusable type {arg_type!r}")
            enum = spec.get("enum")
            arguments.append(
                Argument(
                    name=arg_name,
                    type=arg_type,
                    # Absent means NOT required. The direction is deliberate: a
                    # forgotten flag makes an argument optional, which produces
                    # a corpus case that is merely under-constrained, rather
                    # than required, which would silently start demanding
                    # evidence for something the app does not need.
                    required=bool(spec.get("required", False)),
                    enum=tuple(enum) if isinstance(enum, list) else None,
                )
            )
        tools.append(Tool(name=name, arguments=tuple(arguments)))

    if not tools:
        raise ValueError(f"{path}: declares no tools")

    return ToolSchema(
        version=version,
        status=doc.get("status", "active"),
        tools=tuple(tools),
    )
