"""Scoring one engine's answers against the eval corpus.

Nothing here loads a model, imports a runtime or touches a network. It takes
:class:`EngineResponse` objects — whatever produced them — and turns them into
per-case verdicts, aggregate metrics and a report. That is what makes the
scoring testable without a GPU, and it is why the only module that imports
``needle`` is ``vesta_router.engines.needle``.

FIVE OUTCOME CLASSES, NOT A BOOLEAN

A pass/fail column cannot tell "the router said nothing" from "the router
proposed something the engine withheld" from "the engine crashed", and those
three are different findings about a build. So every case carries two
classifications:

``raw_outcome``      what the engine produced, before anyone screened it:
                     ``tool_call``, ``no_call``, ``suppressed_only``,
                     ``malformed``, ``engine_error``.
``guarded_outcome``  what would have reached dispatch after the evaluator-side
                     grounding screen: ``executable_action``,
                     ``no_executable_action``, ``malformed``, ``engine_error``.

RAW IS NOT HIDDEN BEHIND GUARDED

For "Wake me up." an engine may invent ``time: "07:00"`` and a deterministic
gate may refuse it. Both facts are recorded: the fabrication rate and the final
execution rate are separate numbers, because a router that leans on the guard
every time will be wrong the day the guard has a gap.

ABSENT IS A VALUE

A metric whose denominator is zero is **omitted** from the report, never
emitted as ``0``. On a false-positive-rate row ``0`` reads as perfect, and
``vesta_router.gates`` fails a gate whose metric is missing — which is the
behaviour that makes omission safe rather than convenient.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .corpus import Case
from .grounding import CallVerdict, screen_call
from .render import render_example
from .schema import ToolSchema

__all__ = [
    "Call",
    "CaseResult",
    "EngineResponse",
    "GUARDED_OUTCOMES",
    "RAW_OUTCOMES",
    "aggregate",
    "score_case",
]

#: What the engine produced, before anything screened it.
RAW_OUTCOMES = ("tool_call", "no_call", "suppressed_only", "malformed", "engine_error")

#: What would have reached dispatch.
GUARDED_OUTCOMES = ("executable_action", "no_executable_action", "malformed", "engine_error")

#: Timer durations are compared at the precision the dispatcher actually has.
#: ``apps/mobile/lib/native/system-actions.ts`` computes ``Math.round(minutes *
#: 60)``, so two ``minutes`` values that round to the same whole second are the
#: same timer. Rounding anywhere else would be inventing a tolerance; this one
#: is read off the production dispatcher.
_SECONDS_PER_MINUTE = 60


@dataclass(frozen=True)
class Call:
    """One proposed tool call, as the engine wrote it."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def as_json(self) -> dict:
        return {"name": self.name, "arguments": dict(self.arguments)}


@dataclass(frozen=True)
class EngineResponse:
    """One engine answer. Engine-agnostic on purpose.

    ``error`` set means the runtime failed for this case and nothing else in
    the object is meaningful. Every performance field is ``None`` when the
    runtime did not report it — never ``0``, which on a throughput row reads as
    a measurement.
    """

    raw: dict | None = None
    function_calls: tuple[Call, ...] = ()
    suppressed_calls: tuple[Call, ...] = ()
    reasoning: str | None = None
    latency_ms: float | None = None
    prefill_tps: float | None = None
    decode_tps: float | None = None
    peak_ram_mb: float | None = None
    error: str | None = None

    @property
    def failed(self) -> bool:
        return self.error is not None


@dataclass
class CaseResult:
    case_id: str
    utterance: str
    rendered_prompt: str
    locale: str
    tags: tuple[str, ...]
    source_file: str
    source_line: int

    expected_kind: str
    expected_tool: str | None
    expected_arguments: dict | None
    expected_missing: tuple[str, ...]

    raw_outcome: str
    guarded_outcome: str

    response: EngineResponse
    malformed_reasons: tuple[str, ...] = ()
    verdicts: tuple[CallVerdict, ...] = ()

    # ── classifications ────────────────────────────────────────────────
    expects_action: bool = False
    correct_tool: bool = False
    wrong_tool: bool = False
    extra_tool: bool = False
    no_tool_when_expected: bool = False
    tool_when_none_expected: bool = False
    arguments_exact: bool = False
    full_call_exact: bool = False
    missing_expected_arguments: tuple[str, ...] = ()
    unexpected_extra_arguments: tuple[str, ...] = ()
    wrong_valued_arguments: tuple[str, ...] = ()
    executable_action: bool = False
    fabricated_required_value: bool = False
    placeholder_required_value: bool = False
    stale_context_misuse: tuple[str, ...] = ()
    unit_normalization_correct: bool | None = None
    reset_ok: bool = True
    state_leak: bool = False
    nondeterministic: bool = False

    def as_json(self) -> dict:
        response = self.response
        return {
            "id": self.case_id,
            "utterance": self.utterance,
            "renderedPrompt": self.rendered_prompt,
            "locale": self.locale,
            "tags": list(self.tags),
            "source": {"file": self.source_file, "line": self.source_line},
            "expected": {
                "kind": self.expected_kind,
                "tool": self.expected_tool,
                "arguments": self.expected_arguments,
                "missing": list(self.expected_missing),
            },
            "rawOutcome": self.raw_outcome,
            "guardedOutcome": self.guarded_outcome,
            "engine": {
                "rawEnvelope": response.raw,
                "functionCalls": [c.as_json() for c in response.function_calls],
                "suppressedCalls": [c.as_json() for c in response.suppressed_calls],
                "reasoning": response.reasoning,
                "error": response.error,
                "latencyMs": _round(response.latency_ms, 3),
                "prefillTps": _round(response.prefill_tps, 3),
                "decodeTps": _round(response.decode_tps, 3),
                "peakRamMb": _round(response.peak_ram_mb, 3),
            },
            "malformedReasons": list(self.malformed_reasons),
            "screened": [
                {
                    "tool": v.tool,
                    "admitted": v.admitted,
                    "reason": v.reason,
                    "requiredArguments": [
                        {"name": a.name, "state": a.state, "evidence": a.evidence}
                        for a in v.required_arguments
                    ],
                }
                for v in self.verdicts
            ],
            "classification": {
                "expectsAction": self.expects_action,
                "correctTool": self.correct_tool,
                "wrongTool": self.wrong_tool,
                "extraTool": self.extra_tool,
                "noToolWhenExpected": self.no_tool_when_expected,
                "toolWhenNoneExpected": self.tool_when_none_expected,
                "argumentsExact": self.arguments_exact,
                "fullCallExact": self.full_call_exact,
                "missingExpectedArguments": list(self.missing_expected_arguments),
                "unexpectedExtraArguments": list(self.unexpected_extra_arguments),
                "wrongValuedArguments": list(self.wrong_valued_arguments),
                "executableAction": self.executable_action,
                "fabricatedRequiredValue": self.fabricated_required_value,
                "placeholderRequiredValue": self.placeholder_required_value,
                "staleContextMisuse": list(self.stale_context_misuse),
                "unitNormalizationCorrect": self.unit_normalization_correct,
                "resetOk": self.reset_ok,
                "stateLeak": self.state_leak,
                "nondeterministic": self.nondeterministic,
            },
        }


def _round(value: float | None, places: int) -> float | None:
    return None if value is None else round(float(value), places)


# ── value comparison ────────────────────────────────────────────────────────


def _fold(text: str) -> str:
    return " ".join(str(text).replace("’", "'").casefold().split()).strip(" .!?,;:")


def values_match(tool: str, argument: str, expected: object, actual: object) -> bool:
    """Whether an argument value is the one the corpus asked for.

    Strings compare case-insensitively with whitespace collapsed and trailing
    sentence punctuation dropped: the corpus keeps the user's own words, and a
    trailing full stop is not a routing defect. Everything else about a string
    is compared exactly — a paraphrase IS a defect.

    ``set_timer.minutes`` compares at whole-second dispatch equivalence,
    because that is the precision the production dispatcher has. No other
    rounding happens anywhere.
    """
    if isinstance(expected, bool) or isinstance(actual, bool):
        return expected is actual
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if tool == "set_timer" and argument == "minutes":
            return round(float(expected) * _SECONDS_PER_MINUTE) == round(float(actual) * _SECONDS_PER_MINUTE)
        return float(expected) == float(actual)
    if isinstance(expected, str) and isinstance(actual, str):
        return _fold(expected) == _fold(actual)
    return expected == actual


# ── structural validity ─────────────────────────────────────────────────────


def malformed_reasons(calls: tuple[Call, ...], schema: ToolSchema) -> tuple[str, ...]:
    """Violations of what the decode grammar is supposed to guarantee.

    Needle compiles a byte-level grammar from the schemas it is given and
    claims well-formed output is impossible to violate. So anything here is a
    finding about the ENGINE, not a quality score — which is why
    ``malformedOutputCount`` is the one gate in ``eval/thresholds.json`` that
    is not marked provisional.

    An optional argument the corpus did not want is NOT malformed. It is a
    declared argument of a declared tool, and it is scored as an unexpected
    extra instead.
    """
    reasons: list[str] = []
    for call in calls:
        if not call.name:
            reasons.append("a call has no tool name")
            continue
        tool = schema.tool(call.name)
        if tool is None:
            reasons.append(f"tool {call.name!r} is not in the schema the engine was given")
            continue
        if not isinstance(call.arguments, dict):
            reasons.append(f"{call.name}: arguments is not an object")
            continue
        for name, value in call.arguments.items():
            spec = tool.argument(name)
            if spec is None:
                reasons.append(f"{call.name}: {name!r} is not an argument of this tool")
                continue
            if spec.type == "number" and (isinstance(value, bool) or not isinstance(value, (int, float))):
                reasons.append(f"{call.name}.{name}: declared number, got {type(value).__name__}")
            elif spec.type == "integer" and (isinstance(value, bool) or not isinstance(value, int)):
                reasons.append(f"{call.name}.{name}: declared integer, got {type(value).__name__}")
            elif spec.type == "string" and not isinstance(value, str):
                reasons.append(f"{call.name}.{name}: declared string, got {type(value).__name__}")
            elif spec.enum and isinstance(value, str) and value not in spec.enum:
                reasons.append(f"{call.name}.{name}: {value!r} is not one of {', '.join(spec.enum)}")
    return tuple(reasons)


# ── one case ────────────────────────────────────────────────────────────────


def _pending_known(case: Case) -> dict:
    pending = (case.state or {}).get("pending") or {}
    known = pending.get("known")
    return known if isinstance(known, dict) else {}


def _prior_turn_values(case: Case) -> dict[str, list[object]]:
    """Argument values that appear only in the history block, by argument name.

    Reaching into one of these for a slot the current utterance left empty is
    the spike's finding 4, and ``data/eval/stale-context.jsonl`` is the corpus
    of it.
    """
    values: dict[str, list[object]] = {}
    for turn in (case.state or {}).get("priorTurns") or []:
        for name, value in (turn.get("arguments") or {}).items():
            values.setdefault(name, []).append(value)
    return values


def score_case(case: Case, response: EngineResponse, schema: ToolSchema) -> CaseResult:
    """Classify and score one engine answer. No I/O, no engine, no clock."""
    expected = case.outcome
    kind = expected.get("kind", "")
    expected_tool = expected.get("tool") if kind != "chat" else None
    expected_arguments = expected.get("arguments") if kind == "tool" else None
    expected_missing = tuple(expected.get("missing") or ()) if kind == "incomplete" else ()

    result = CaseResult(
        case_id=case.id,
        utterance=case.utterance,
        rendered_prompt=render_example(case.utterance, case.state, schema),
        locale=case.locale,
        tags=case.tags,
        source_file=case.file,
        source_line=case.line,
        expected_kind=kind,
        expected_tool=expected_tool,
        expected_arguments=expected_arguments,
        expected_missing=expected_missing,
        raw_outcome="engine_error",
        guarded_outcome="engine_error",
        response=response,
        expects_action=kind == "tool",
    )

    if response.failed:
        return result

    emitted = response.function_calls
    suppressed = response.suppressed_calls
    result.malformed_reasons = malformed_reasons(emitted + suppressed, schema)

    if result.malformed_reasons:
        result.raw_outcome = "malformed"
        result.guarded_outcome = "malformed"
    elif emitted:
        result.raw_outcome = "tool_call"
    elif suppressed:
        result.raw_outcome = "suppressed_only"
    else:
        result.raw_outcome = "no_call"

    # ── the guardrail pass ─────────────────────────────────────────────
    # Screened against the user's own words only. The rendered prompt is what
    # the model saw; the utterance plus the carried pending state is what the
    # user actually supplied, and a history block is neither.
    known = _pending_known(case)
    result.verdicts = tuple(
        screen_call(call.name, call.arguments, case.utterance, schema, known) for call in emitted
    )
    result.executable_action = any(v.admitted for v in result.verdicts)
    if result.guarded_outcome != "malformed":
        result.guarded_outcome = "executable_action" if result.executable_action else "no_executable_action"

    # ── tool selection ─────────────────────────────────────────────────
    names = [c.name for c in emitted]
    if kind == "tool":
        result.correct_tool = names == [expected_tool]
        result.wrong_tool = bool(names) and expected_tool not in names
        result.extra_tool = len(names) > 1
        result.no_tool_when_expected = not names
    else:
        result.tool_when_none_expected = bool(names)

    # ── arguments ──────────────────────────────────────────────────────
    if kind == "tool" and expected_tool in names and isinstance(expected_arguments, dict):
        call = next(c for c in emitted if c.name == expected_tool)
        actual = call.arguments if isinstance(call.arguments, dict) else {}
        result.missing_expected_arguments = tuple(name for name in expected_arguments if name not in actual)
        result.unexpected_extra_arguments = tuple(name for name in actual if name not in expected_arguments)
        result.wrong_valued_arguments = tuple(
            name
            for name, value in expected_arguments.items()
            if name in actual and not values_match(expected_tool, name, value, actual[name])
        )
        result.arguments_exact = not (
            result.missing_expected_arguments
            or result.unexpected_extra_arguments
            or result.wrong_valued_arguments
        )
        result.full_call_exact = result.correct_tool and result.arguments_exact

    # ── fabrication of a required slot the user never gave ─────────────
    # Counted over EMITTED AND SUPPRESSED calls alike: a proposal the engine's
    # own gate withheld is still a proposal, and the spike showed that gate
    # letting an identical invention through on the next prompt.
    if kind in ("incomplete", "chat", "cancel"):
        for call in emitted + suppressed:
            tool = schema.tool(call.name)
            if tool is None:
                continue
            verdict = screen_call(call.name, call.arguments, case.utterance, schema, known)
            for argument in verdict.required_arguments:
                if argument.state == "ungrounded":
                    result.fabricated_required_value = True
                elif argument.state == "missing" and argument.name in (call.arguments or {}):
                    # Present in the JSON but empty or non-dispatchable: `""`,
                    # `minutes: 0`. Not an invention, and not an action either.
                    result.placeholder_required_value = True

    # ── stale context ──────────────────────────────────────────────────
    prior = _prior_turn_values(case)
    if prior:
        misuse: list[str] = []
        for call in emitted + suppressed:
            verdict = screen_call(call.name, call.arguments, case.utterance, schema, known)
            ungrounded = set(verdict.ungrounded)
            for name, value in (call.arguments or {}).items():
                if name not in prior:
                    continue
                if not any(values_match(call.name, name, old, value) for old in prior[name]):
                    continue
                # An optional argument has no grounding verdict of its own, so
                # carrying one forward (a date from the previous turn) is
                # judged by whether the utterance repeated it.
                if name in ungrounded or not _in_utterance(value, case.utterance):
                    misuse.append(f"{call.name}.{name}")
        result.stale_context_misuse = tuple(sorted(set(misuse)))

    # ── unit normalization ─────────────────────────────────────────────
    if "unit-conversion" in case.tags and kind == "tool":
        result.unit_normalization_correct = result.correct_tool and not result.wrong_valued_arguments

    return result


def _in_utterance(value: object, utterance: str) -> bool:
    return _fold(str(value)) in _fold(utterance)


# ── aggregation ─────────────────────────────────────────────────────────────


def _rate(numerator: int, denominator: int) -> float | None:
    """A rate, or ``None`` when nothing was measured.

    ``None`` is what keeps the metric out of the report entirely. Zero cases
    over zero cases is not 0.0 and is not 1.0; it is a question nobody asked.
    """
    if denominator <= 0:
        return None
    return round(numerator / denominator, 6)


def _percentile(values: list[float], fraction: float) -> float | None:
    """Nearest-rank percentile: the smallest value at or above the fraction.

    Nearest-rank rather than interpolated so a reported p95 is a latency that
    actually happened, not an average of two that did not.
    """
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return round(ordered[index], 3)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return round(ordered[middle], 3)
    return round((ordered[middle - 1] + ordered[middle]) / 2, 3)


def _metrics_for(results: list[CaseResult]) -> dict:
    """Every metric a set of case results supports. Absent stays absent."""
    action = [r for r in results if r.expected_kind == "tool"]
    chat = [r for r in results if r.expected_kind == "chat"]
    incomplete = [r for r in results if r.expected_kind == "incomplete"]
    cancel = [r for r in results if r.expected_kind == "cancel"]
    units = [r for r in results if r.unit_normalization_correct is not None]
    correct_tool = [r for r in action if r.correct_tool]

    metrics: dict[str, float | int] = {}

    def put(key: str, value: float | None) -> None:
        if value is not None:
            metrics[key] = value

    put("toolSelectionAccuracy", _rate(sum(r.correct_tool for r in action), len(action)))
    put("argumentExactMatchAccuracy", _rate(sum(r.arguments_exact for r in correct_tool), len(correct_tool)))
    put("fullCallExactMatchAccuracy", _rate(sum(r.full_call_exact for r in action), len(action)))
    put("noToolClassificationAccuracy", _rate(sum(not r.tool_when_none_expected for r in chat), len(chat)))
    put("falsePositiveToolCallRate", _rate(sum(r.tool_when_none_expected for r in chat), len(chat)))
    put("falsePositiveActionRate", _rate(sum(r.executable_action for r in chat), len(chat)))
    put(
        "missingSlotFabricationRate",
        _rate(sum(r.fabricated_required_value for r in incomplete), len(incomplete)),
    )
    put(
        "missingSlotPlaceholderRate",
        _rate(sum(r.placeholder_required_value for r in incomplete), len(incomplete)),
    )
    put(
        "missingSlotEmissionRate",
        _rate(sum(r.raw_outcome == "tool_call" for r in incomplete), len(incomplete)),
    )
    put("missingSlotExecutionRate", _rate(sum(r.executable_action for r in incomplete), len(incomplete)))
    put("cancelNoActionRate", _rate(sum(not r.executable_action for r in cancel), len(cancel)))
    put(
        "unitNormalizationAccuracy", _rate(sum(bool(r.unit_normalization_correct) for r in units), len(units))
    )

    metrics["malformedOutputCount"] = sum(1 for r in results if r.raw_outcome == "malformed")
    metrics["engineErrorCount"] = sum(1 for r in results if r.raw_outcome == "engine_error")
    metrics["staleContextMisuseCount"] = sum(1 for r in results if r.stale_context_misuse)
    metrics["stateLeakCount"] = sum(1 for r in results if r.state_leak)
    metrics["nondeterministicCaseCount"] = sum(1 for r in results if r.nondeterministic)
    metrics["resetFailureCount"] = sum(1 for r in results if not r.reset_ok)
    return metrics


def _performance(results: list[CaseResult], wall_clock_ms: float | None) -> dict:
    latencies = [r.response.latency_ms for r in results if r.response.latency_ms is not None]
    prefill = [r.response.prefill_tps for r in results if r.response.prefill_tps is not None]
    decode = [r.response.decode_tps for r in results if r.response.decode_tps is not None]
    ram = [r.response.peak_ram_mb for r in results if r.response.peak_ram_mb is not None]

    performance: dict[str, float] = {}

    def put(key: str, value: float | None) -> None:
        if value is not None:
            performance[key] = value

    put("medianLatencyMs", _median(latencies))
    put("p95LatencyMs", _percentile(latencies, 0.95))
    put("minLatencyMs", round(min(latencies), 3) if latencies else None)
    put("maxLatencyMs", round(max(latencies), 3) if latencies else None)
    put("totalWallClockMs", None if wall_clock_ms is None else round(wall_clock_ms, 3))
    put("medianPrefillTps", _median(prefill))
    put("medianDecodeTps", _median(decode))
    # The engine reports a per-call peak for its own process; the maximum over
    # the run is the only honest summary of a series of peaks.
    put("peakMemoryMb", round(max(ram), 3) if ram else None)
    return performance


def aggregate(
    results: list[CaseResult],
    wall_clock_ms: float | None = None,
    tags: tuple[str, ...] = (),
) -> dict:
    """Overall metrics, performance and a per-tag breakdown.

    Per-tag exists so a regression is localizable. A tag with no cases produces
    no entry, for the same reason a rate with no denominator produces no
    number.
    """
    per_tag: dict[str, dict] = {}
    wanted = tags or tuple(sorted({tag for r in results for tag in r.tags}))
    for tag in wanted:
        subset = [r for r in results if tag in r.tags]
        if not subset:
            continue
        entry = {"cases": len(subset)}
        entry.update(_metrics_for(subset))
        per_tag[tag] = entry

    return {
        "cases": len(results),
        "metrics": _metrics_for(results),
        "performance": _performance(results, wall_clock_ms),
        "perTag": per_tag,
        "outcomeCounts": {
            "raw": {name: sum(1 for r in results if r.raw_outcome == name) for name in RAW_OUTCOMES},
            "guarded": {
                name: sum(1 for r in results if r.guarded_outcome == name) for name in GUARDED_OUTCOMES
            },
        },
    }
