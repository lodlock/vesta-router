"""Engines the evaluator can drive, and the boundary they sit behind.

An engine has four operations and no opinions about scoring:

``describe()``  what ran — engine name, version, artifact digest. Goes into the
                report so a number is attributable to a file.
``reset()``     drop conversation state. Returns whether it worked, because a
                reset that silently failed is how one case's values end up in
                the next case's answer.
``complete()``  one rendered prompt in, one :class:`EngineResponse` out.
``close()``     release the runtime.

Everything in this module is standard library. The only module in the package
that imports a third-party runtime is :mod:`vesta_router.engines.needle`, and
it does so inside the function that needs it — so ``python -m vesta_router
validate`` still runs on a laptop with nothing installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from ..evaluate import Call, EngineResponse

__all__ = ["Engine", "ReplayEngine", "response_from_envelope", "tool_definitions"]


class Engine(Protocol):
    def describe(self) -> dict: ...

    def reset(self) -> bool: ...

    def complete(self, prompt: str) -> EngineResponse: ...

    def close(self) -> None: ...


def tool_definitions(schema_path: str | Path) -> list[dict]:
    """Translate a Vesta tool schema into the flat form an engine is given.

    ``{name, description, parameters}`` with a JSON-Schema object for
    ``parameters`` — the shape ``cactus-needle``'s own ``build_schema()``
    produces, which is also what the Android spike sent over JNI
    (``apps/mobile/lib/router-spike/tools.ts``). Same surface on the host and
    on the device, so a host baseline is comparable with a device run.

    Ordering is the schema's own, and ``required`` follows the declared
    argument order rather than insertion chance, so the JSON handed to an
    engine is byte-stable across runs. The engine tokenizes it into a static
    prefix; a reordering would change the prefix and therefore the run.
    """
    doc = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    definitions: list[dict] = []
    for entry in doc.get("tools", []):
        properties: dict[str, dict] = {}
        required: list[str] = []
        for name, spec in (entry.get("parameters") or {}).items():
            prop: dict = {"type": spec["type"]}
            if spec.get("description"):
                prop["description"] = spec["description"]
            if spec.get("enum"):
                prop["enum"] = list(spec["enum"])
            properties[name] = prop
            if spec.get("required"):
                required.append(name)
        parameters: dict = {"type": "object", "properties": properties}
        if required:
            parameters["required"] = required
        definitions.append(
            {
                "name": entry["name"],
                "description": entry.get("description", ""),
                "parameters": parameters,
            }
        )
    return definitions


def _calls(raw: object) -> tuple[Call, ...]:
    if not isinstance(raw, list):
        return ()
    calls = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        arguments = entry.get("arguments")
        calls.append(
            Call(
                name=str(entry.get("name", "")),
                arguments=arguments if isinstance(arguments, dict) else {},
            )
        )
    return tuple(calls)


def _number(raw: object) -> float | None:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    return float(raw)


def response_from_envelope(envelope: dict, latency_ms: float | None = None) -> EngineResponse:
    """Read one engine envelope into the evaluator's own shape.

    A field the engine did not report stays ``None``. Nothing is defaulted to
    zero here: ``peak_ram_mb = 0`` would be a measurement, and there was none.
    """
    error = None
    if envelope.get("success") is False:
        error = str(envelope.get("error") or envelope.get("reason") or "engine reported failure")
    return EngineResponse(
        raw=envelope,
        function_calls=_calls(envelope.get("function_calls")),
        suppressed_calls=_calls(envelope.get("suppressed_calls")),
        reasoning=envelope.get("reasoning") if isinstance(envelope.get("reasoning"), str) else None,
        latency_ms=latency_ms,
        prefill_tps=_number(envelope.get("prefill_tps")),
        decode_tps=_number(envelope.get("decode_tps")),
        peak_ram_mb=_number(envelope.get("peak_ram_mb")),
        error=error,
    )


class ReplayEngine:
    """An engine that replays recorded envelopes instead of running a model.

    It exists so the scoring, the classification and the report can be tested
    in CI without a 34 MB artifact and without a runtime — and it is NEVER a
    substitute for a baseline. A report it produces carries
    ``engine: "replay"`` and no artifact digest, and nothing downstream will
    mistake that for a measurement of a model.
    """

    def __init__(self, envelopes: dict[str, dict], latency_ms: float | None = None) -> None:
        self._envelopes = dict(envelopes)
        self._latency_ms = latency_ms
        self.resets = 0
        self.prompts: list[str] = []

    def describe(self) -> dict:
        return {"engine": "replay", "engineVersion": None, "artifactSha256": None}

    def reset(self) -> bool:
        self.resets += 1
        return True

    def complete(self, prompt: str) -> EngineResponse:
        self.prompts.append(prompt)
        envelope = self._envelopes.get(prompt)
        if envelope is None:
            return EngineResponse(error=f"no recorded envelope for prompt {prompt!r}")
        return response_from_envelope(envelope, self._latency_ms)

    def close(self) -> None:
        return None
