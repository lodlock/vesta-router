"""Running the eval corpus against an engine, and writing the report.

RESET BETWEEN EVERY CASE, AND A SECOND PASS THAT CHECKS IT

Vesta owns conversation state. A corpus case therefore carries *all* the state
it is allowed to see, in its own ``state`` block, and the engine must arrive at
it holding nothing. So every case is preceded by ``reset()``, and a failed
reset is recorded rather than shrugged off.

That is the policy. Checking whether it *worked* needs a second pass, because
an engine that silently ignores a reset produces perfectly plausible answers —
the spike's finding 4 was exactly that: before ``needle_reset`` was called
between prompts, under-specified prompts came back filled in from earlier
turns, and the reasoning text said so.

The isolation pass is a **fresh engine process, the corpus in reverse order,
and every case run twice in a row**:

* two consecutive answers that differ  → ``nondeterministic``. The engine is not
  reproducible and neither is anything below it, so no leak claim is made.
* two that agree but differ from pass A → ``state_leak``. The answer depended on
  what ran before it, which is the thing reset is supposed to prevent.

Reverse order matters: same order with a fresh process would reproduce the same
neighbours and see nothing.

METRICS COME FROM PASS A. The isolation pass contributes ``stateLeakCount`` and
``nondeterministicCaseCount`` and nothing else — scoring a case twice and
averaging would report a build nobody can reproduce.
"""

from __future__ import annotations

import datetime as _datetime
import hashlib
import json
import platform
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from .corpus import Case
from .engines import Engine
from .evaluate import CaseResult, aggregate, score_case
from .render import render_example
from .schema import ToolSchema

__all__ = [
    "EVALUATOR_VERSION",
    "RunOutcome",
    "baseline_identifier",
    "corpus_digest",
    "run_evaluation",
    "write_report",
]

#: Bumped when the SCORING changes, not when the harness is refactored. Two
#: reports carrying different evaluator versions are not comparable, and the
#: number is in every report so that is visible rather than assumed.
EVALUATOR_VERSION = "1.0.0"


@dataclass
class RunOutcome:
    results: list[CaseResult]
    wall_clock_ms: float
    engine: dict
    isolation_ran: bool


def _fingerprint(result_or_response) -> str:
    """The part of an answer that must be stable, as canonical JSON.

    Latency, throughput and peak RAM are excluded: they vary run to run and
    comparing them would report noise as a leak.
    """
    response = getattr(result_or_response, "response", result_or_response)
    return json.dumps(
        {
            "functionCalls": [c.as_json() for c in response.function_calls],
            "suppressedCalls": [c.as_json() for c in response.suppressed_calls],
            "reasoning": response.reasoning,
            "error": response.error,
        },
        sort_keys=True,
        ensure_ascii=False,
    )


def run_evaluation(
    engine_factory: Callable[[], Engine],
    cases: list[Case],
    schema: ToolSchema,
    isolation_check: bool = True,
    on_case: Callable[[CaseResult], None] | None = None,
) -> RunOutcome:
    """Score every case, then optionally check that the resets held."""
    import time

    engine = engine_factory()
    description = engine.describe()
    results: list[CaseResult] = []
    started = time.perf_counter()
    try:
        for case in cases:
            reset_ok = engine.reset()
            prompt = render_example(case.utterance, case.state, schema)
            response = engine.complete(prompt)
            result = score_case(case, response, schema)
            result.reset_ok = reset_ok
            results.append(result)
            if on_case is not None:
                on_case(result)
    finally:
        engine.close()
    wall_clock_ms = (time.perf_counter() - started) * 1000

    if isolation_check:
        _check_isolation(engine_factory, cases, schema, results)

    return RunOutcome(
        results=results,
        wall_clock_ms=wall_clock_ms,
        engine=description,
        isolation_ran=isolation_check,
    )


def _check_isolation(
    engine_factory: Callable[[], Engine],
    cases: list[Case],
    schema: ToolSchema,
    results: list[CaseResult],
) -> None:
    by_id = {result.case_id: result for result in results}
    engine = engine_factory()
    try:
        for case in reversed(cases):
            result = by_id.get(case.id)
            if result is None:
                continue
            prompt = render_example(case.utterance, case.state, schema)
            engine.reset()
            first = engine.complete(prompt)
            engine.reset()
            second = engine.complete(prompt)
            if _fingerprint(first) != _fingerprint(second):
                result.nondeterministic = True
            elif _fingerprint(first) != _fingerprint(result):
                result.state_leak = True
    finally:
        engine.close()


# ── identity and provenance ─────────────────────────────────────────────────


def baseline_identifier(artifact_path: str | Path, sha256: str) -> str:
    """A name for a benchmark result, deliberately unlike a model version.

    ``baseline-needle3-c9d915eca282`` names a measurement of one file. It is
    not a ``ModelVersion``, it never becomes one, and evaluating a model does
    not make it ``stable`` — promotion is a decision about a trained candidate
    against the gates, and this is the number those gates will one day be set
    from.
    """
    return f"baseline-{Path(artifact_path).stem}-{sha256[:12]}"


def corpus_digest(paths: Iterable[Path]) -> dict:
    """A content digest of the corpus, per file and overall.

    Content rather than a git revision: a report must stay attributable when it
    is produced from a dirty tree, which during harness development is most of
    the time. A git sha would be a claim about a commit that may not contain
    the bytes that were scored.
    """
    files = []
    overall = hashlib.sha256()
    for path in sorted(paths, key=lambda p: p.name):
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        files.append({"name": path.name, "sha256": digest, "sizeBytes": len(data)})
        overall.update(f"{path.name}:{digest}\n".encode())
    return {"files": files, "sha256": overall.hexdigest()}


def _utc_now() -> _datetime.datetime:
    return _datetime.datetime.now(_datetime.timezone.utc).replace(microsecond=0)


def write_report(
    output_root: str | Path,
    outcome: RunOutcome,
    schema: ToolSchema,
    schema_path: Path,
    corpus_paths: Iterable[Path],
    baseline_id: str,
    notes: Iterable[str] = (),
    limited: bool = False,
) -> Path:
    """Write ``summary.json``, ``cases.jsonl`` and ``manifest.json``.

    Deterministic serialization — sorted keys, one case per line, no trailing
    whitespace — so two reports of the same run diff to nothing but their
    timestamps and their latencies.
    """
    now = _utc_now()
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    directory = Path(output_root) / f"{baseline_id}-{stamp}"
    directory.mkdir(parents=True, exist_ok=True)

    corpus = corpus_digest(corpus_paths)
    summary_stats = aggregate(outcome.results, outcome.wall_clock_ms)
    engine = dict(outcome.engine)

    provenance = {
        "baselineId": baseline_id,
        "evaluatorVersion": EVALUATOR_VERSION,
        "artifactSha256": engine.get("artifactSha256"),
        "toolSchemaVersion": schema.version,
        "corpusContentSha256": corpus["sha256"],
        "engineVersion": engine.get("engineVersion"),
    }

    summary = {
        "reportVersion": 1,
        "kind": "baseline",
        "evaluatedAt": now.isoformat().replace("+00:00", "Z"),
        "modelVersion": None,
        "artifactKind": engine.get("artifactKind"),
        "cases": summary_stats["cases"],
        "partialRun": limited,
        "metrics": summary_stats["metrics"],
        "performance": summary_stats["performance"],
        "outcomeCounts": summary_stats["outcomeCounts"],
        "perTag": summary_stats["perTag"],
        "notes": list(notes),
        **provenance,
    }

    manifest = {
        "reportVersion": 1,
        "kind": "baseline",
        "generatedAt": summary["evaluatedAt"],
        "baselineId": baseline_id,
        "evaluatorVersion": EVALUATOR_VERSION,
        "artifact": {
            "path": engine.get("artifactPath"),
            "sha256": engine.get("artifactSha256"),
            "sizeBytes": engine.get("artifactSizeBytes"),
            "kind": engine.get("artifactKind"),
        },
        "runtime": {
            "engine": engine.get("engine"),
            "engineVersion": engine.get("engineVersion"),
            "engineLibrary": engine.get("engineLibrary"),
        },
        "prompt": {
            "systemPrompt": engine.get("systemPrompt"),
            "autoDate": engine.get("autoDate"),
            "renderer": "vesta_router.render.render_example",
        },
        "toolSchema": {
            "version": schema.version,
            "path": str(schema_path),
            "sha256": hashlib.sha256(schema_path.read_bytes()).hexdigest(),
        },
        "corpus": corpus,
        "policy": {
            "resetBetweenCases": True,
            "isolationPass": outcome.isolation_ran,
            "isolationMethod": (
                "fresh engine process, corpus reversed, each case run twice; "
                "a differing pair is nondeterminism, an agreeing pair that "
                "differs from the scored pass is a state leak"
            ),
            "metricsFrom": "the first pass only",
            "guardrail": (
                "vesta_router.grounding — an evaluator-side approximation of Vesta's "
                "deterministic screen, NOT the app's own code"
            ),
            "partialRun": limited,
        },
        "host": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
    }

    _write_json(directory / "summary.json", summary)
    _write_json(directory / "manifest.json", manifest)
    with (directory / "cases.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for result in outcome.results:
            payload = dict(result.as_json())
            payload["provenance"] = provenance
            handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=False))
            handle.write("\n")

    return directory


def _write_json(path: Path, payload: dict) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
    path.write_text(text + "\n", encoding="utf-8", newline="\n")
