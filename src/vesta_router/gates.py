"""Promotion gates.

A gate is a threshold an eval result must clear before a build may be promoted
to ``stable``. They are read from ``eval/thresholds.json``.

JSON, not YAML, and the reason is not taste: the fast CI tier must run with the
standard library alone so it can run anywhere, including on a laptop with no
environment set up. Training configs stay YAML — they are read by the training
toolchain, which has plenty of dependencies already.

THE ASYMMETRY IS THE POINT

Safety metrics are not traded against accuracy. A false executable action costs
the user something real — a wrong alarm, a call that has already rung, a message
sent under their name. An unnecessary escalation to the chat model costs a
little latency. So ``missingSlotExecutionRate`` and ``staleContextMisuseCount``
are gated at zero and a gain in tool-selection accuracy does not buy a
relaxation.

EVERY THRESHOLD IS PROVISIONAL

No model has been trained, so no number here is earned. They are starting points
to be replaced by measured ones once a baseline exists, and
``thresholds.json`` says so in its own text. A gate whose metric is ABSENT from
the report fails: a metric nobody measured is not a metric that passed, and
treating a missing number as a pass is how an unmeasured build gets promoted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Gate", "GateResult", "load_gates", "apply_gates"]

_COMPARATORS = {
    "<=": lambda actual, limit: actual <= limit,
    "<": lambda actual, limit: actual < limit,
    ">=": lambda actual, limit: actual >= limit,
    ">": lambda actual, limit: actual > limit,
    "==": lambda actual, limit: actual == limit,
}


@dataclass(frozen=True)
class Gate:
    metric: str
    comparator: str
    threshold: float
    safety: bool
    provisional: bool
    rationale: str


@dataclass(frozen=True)
class GateResult:
    gate: Gate
    actual: float | None
    passed: bool

    def __str__(self) -> str:
        if self.actual is None:
            requirement = f"{self.gate.comparator} {self.gate.threshold}"
            return f"{self.gate.metric}: NOT MEASURED (gate requires {requirement})"
        verdict = "pass" if self.passed else "FAIL"
        return (
            f"{self.gate.metric}: {self.actual} {self.gate.comparator} {self.gate.threshold} -> {verdict}"
            + (" [safety]" if self.gate.safety else "")
        )


def load_gates(path: str | Path) -> list[Gate]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    gates: list[Gate] = []
    for entry in doc.get("gates", []):
        comparator = entry.get("comparator")
        if comparator not in _COMPARATORS:
            raise ValueError(f"{path}: unusable comparator {comparator!r} for {entry.get('metric')}")
        gates.append(
            Gate(
                metric=entry["metric"],
                comparator=comparator,
                threshold=float(entry["threshold"]),
                safety=bool(entry.get("safety", False)),
                provisional=bool(entry.get("provisional", True)),
                rationale=entry.get("rationale", ""),
            )
        )
    if not gates:
        raise ValueError(f"{path}: declares no gates")
    return gates


def apply_gates(metrics: dict, gates: list[Gate]) -> list[GateResult]:
    results: list[GateResult] = []
    for gate in gates:
        actual = metrics.get(gate.metric)
        if not isinstance(actual, (int, float)) or isinstance(actual, bool):
            # Absent is not a pass. The build was not measured on this, and a
            # promotion that treats "not measured" as "fine" is exactly the
            # failure the manifest's evaluatedArtifactSha256 rule also guards.
            results.append(GateResult(gate=gate, actual=None, passed=False))
            continue
        results.append(
            GateResult(gate=gate, actual=actual, passed=_COMPARATORS[gate.comparator](actual, gate.threshold))
        )
    return results


def promotable(results: list[GateResult]) -> bool:
    return all(r.passed for r in results)
