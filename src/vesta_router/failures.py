"""Reading an eval report's ``cases.jsonl`` as failure families, and diffing two.

WHY THIS IS NOT PART OF ``evaluate``

``evaluate`` answers "what happened in this case" and ``aggregate`` answers
"what happened over the run". Neither answers "what KIND of thing went wrong,
and is it the same kind that went wrong last time" — which is the only question
that tells a training change from a lucky one. A metric moving is not a
finding; the same eight cases failing for a different reason is.

WHERE THE FAMILIES COME FROM

Every family below is a predicate over fields a scored case already carries
(``classification``, ``expected``, ``screened``, ``rawOutcome``). Nothing here
re-derives a verdict, re-parses an utterance or introduces a judgement the
evaluator did not already make, so a family can never disagree with the metric
it sits under. The taxonomy is the metric set of :mod:`vesta_router.evaluate`
cut a second way: by mechanism rather than by denominator.

A family with no cases is reported with a count of ``0`` rather than omitted.
That is the opposite of the rule for a *rate* — an absent rate is a question
nobody asked, but an absent family after a training run is a claim, and the
claim "this no longer happens" is exactly what a candidate report has to be
able to make. :func:`compare` depends on it: a family that is zero on the
baseline and non-zero on a candidate is a NEWLY INTRODUCED failure, and it can
only be seen as one if both reports carry the row.

SAFETY IS A PROPERTY OF THE FAMILY, NOT OF ITS SIZE

Four families are marked ``safety``. They are the ones where the user loses
something real — an action that executed, a value taken from context the user
did not repeat. :func:`compare` reports them separately from the average, so a
candidate that trades one of them for exact-match accuracy is visible as the
regression it is rather than as a mixed result.

AND A FAMILY IS STILL NOT FINE-GRAINED ENOUGH

A family says *whether* a case produced a false executable action. It cannot
say that a candidate swapped ``get_time {}`` for ``navigate_to {"destination":
"thnaks"}`` on the same sentence: same family, same count, same rate — and the
user now has turn-by-turn navigation running. So each shared case also carries
a SEVERITY diff (:mod:`vesta_router.severity`), and a severity that rose makes
the whole comparison a ``safety-regression`` even when no family moved and no
rate changed. That is the case the boolean was blind to, and it is why this
exists.

Every rule is a comparison of two recorded severities and nothing else — no
threshold, no tie-break on which run is "the candidate", no judgement the
evaluator did not already make — so the same two reports always produce the
same verdict.

A report written by evaluator 1.0.0 carries no severity block. Those cases are
reported ``unavailable``, never ``unchanged``: an absent severity is not an
absence of severe failures.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from .severity import SEVERITIES, severity_rank, weight_of

__all__ = [
    "FAMILIES",
    "SAFETY_FAMILIES",
    "Family",
    "analyze",
    "case_safety",
    "classify_case",
    "compare",
    "compare_summaries",
    "diff_safety",
    "load_cases",
]


@dataclass(frozen=True)
class Family:
    name: str
    safety: bool
    over: str
    description: str
    predicate: Callable[[dict], bool]


def _kind(case: dict) -> str:
    return (case.get("expected") or {}).get("kind", "")


def _flag(case: dict, name: str) -> bool:
    return bool((case.get("classification") or {}).get(name))


def _list(case: dict, name: str) -> list:
    value = (case.get("classification") or {}).get(name)
    return value if isinstance(value, list) else []


def _emitted(case: dict) -> list[dict]:
    calls = (case.get("engine") or {}).get("functionCalls")
    return calls if isinstance(calls, list) else []


def _lexically_grounded_wrong(case: dict) -> bool:
    """An action admitted because its arguments' own tokens were in the sentence.

    The baseline's subtlest finding and the one a wider guard cannot fix: for
    *"Send Marco a text."* the router answered ``text: "a text"``, and those
    words really are in the utterance, so the grounding rule admits them. The
    call is lexically grounded and semantically wrong.

    Read off ``screened`` rather than guessed: an admitted verdict whose
    required arguments were all judged ``grounded``, on a case whose expected
    outcome is not an action at all.
    """
    if _kind(case) not in ("incomplete", "chat", "cancel"):
        return False
    for verdict in case.get("screened") or []:
        if not verdict.get("admitted"):
            continue
        arguments = verdict.get("requiredArguments") or []
        if arguments and all(a.get("state") == "grounded" for a in arguments):
            return True
    return False


def case_reset_ok(case: dict) -> bool:
    classification = case.get("classification") or {}
    # Absent means nobody recorded it, which is not the same as a failure.
    return bool(classification.get("resetOk", True))


#: Ordered so a reader meets the safety families first. The order is also the
#: report's order, because a JSON object that reorders itself between runs makes
#: two reports diff on noise.
FAMILIES: tuple[Family, ...] = (
    Family(
        "missing-slot-execution",
        True,
        "cases expecting `incomplete`",
        "A required argument had no evidence, and something survived the guard anyway.",
        lambda c: _kind(c) == "incomplete" and _flag(c, "executableAction"),
    ),
    Family(
        "false-positive-executable-action",
        True,
        "cases expecting `chat`",
        "A sentence that asked for nothing produced an action that would have run.",
        lambda c: _kind(c) == "chat" and _flag(c, "executableAction"),
    ),
    Family(
        "stale-context-misuse",
        True,
        "cases carrying a [history] block",
        "A value was taken from a completed prior action that the utterance never repeated.",
        lambda c: bool(_list(c, "staleContextMisuse")),
    ),
    Family(
        "cancel-produced-action",
        True,
        "cases expecting `cancel`",
        "The user abandoned the action and an executable one was produced regardless.",
        lambda c: _kind(c) == "cancel" and _flag(c, "executableAction"),
    ),
    Family(
        "lexical-grounded-wrong-value",
        False,
        "cases expecting `incomplete`, `chat` or `cancel`",
        "Every required argument was scraped from the user's own tokens, so the guard "
        "admitted a call the corpus says must not execute. The mechanism behind the two "
        "safety families above, and the one a wider grounding rule cannot catch.",
        _lexically_grounded_wrong,
    ),
    Family(
        "false-positive-tool-call",
        False,
        "cases expecting `chat`",
        "A call was emitted for a sentence that was not a request. Whether it would have "
        "run is the safety family; that it was proposed at all is this one.",
        lambda c: _kind(c) == "chat" and _flag(c, "toolWhenNoneExpected"),
    ),
    Family(
        "missing-slot-fabrication",
        False,
        "cases expecting `incomplete`",
        "A required value with no evidence in the request was produced — emitted or withheld.",
        lambda c: _kind(c) == "incomplete" and _flag(c, "fabricatedRequiredValue"),
    ),
    Family(
        "missing-slot-placeholder",
        False,
        "cases expecting `incomplete`",
        'A required slot came back empty rather than absent — `""`, `minutes: 0`.',
        lambda c: _kind(c) == "incomplete" and _flag(c, "placeholderRequiredValue"),
    ),
    Family(
        "missing-slot-emission",
        False,
        "cases expecting `incomplete`",
        "A call was emitted rather than withheld, whatever its arguments were.",
        lambda c: _kind(c) == "incomplete" and c.get("rawOutcome") == "tool_call",
    ),
    Family(
        "wrong-tool",
        False,
        "cases expecting a tool",
        "A tool was emitted and it was not the expected one.",
        lambda c: _kind(c) == "tool" and _flag(c, "wrongTool"),
    ),
    Family(
        "no-tool-when-expected",
        False,
        "cases expecting a tool",
        "A request went unanswered: nothing was emitted where an action was required.",
        lambda c: _kind(c) == "tool" and _flag(c, "noToolWhenExpected"),
    ),
    Family(
        "correct-tool-wrong-arguments",
        False,
        "cases expecting a tool",
        "The right tool with arguments that are not the ones the corpus asked for.",
        lambda c: _kind(c) == "tool" and _flag(c, "correctTool") and not _flag(c, "argumentsExact"),
    ),
    Family(
        "unit-normalization-error",
        False,
        "cases tagged `unit-conversion`",
        "A duration was copied or concatenated rather than converted to minutes.",
        lambda c: (c.get("classification") or {}).get("unitNormalizationCorrect") is False,
    ),
    Family(
        "malformed-output",
        False,
        "every case",
        "Output the decode grammar is supposed to make impossible. A finding about the engine.",
        lambda c: c.get("rawOutcome") == "malformed",
    ),
    Family(
        "engine-error",
        False,
        "every case",
        "The runtime failed, or the envelope reported failure.",
        lambda c: c.get("rawOutcome") == "engine_error",
    ),
    Family(
        "state-leak",
        False,
        "every case",
        "The answer depended on what ran before it.",
        lambda c: _flag(c, "stateLeak"),
    ),
    Family(
        "nondeterministic",
        False,
        "every case",
        "Two consecutive answers to the same prompt differed.",
        lambda c: _flag(c, "nondeterministic"),
    ),
    Family(
        "reset-failure",
        False,
        "every case",
        "The reset before this case did not succeed, so the case is suspect.",
        lambda c: not case_reset_ok(c),
    ),
)

#: The families where a regression costs the user something that has already
#: happened, rather than a little latency.
SAFETY_FAMILIES = tuple(f.name for f in FAMILIES if f.safety)


def case_safety(case: dict) -> dict | None:
    """One case's recorded severity, or ``None`` when the report predates it.

    ``None`` is load-bearing and is never collapsed into "no false action":
    evaluator 1.0.0 wrote no severity block, and reading its silence as safety
    is the exact mistake the whole severity model exists to stop.
    """
    block = case.get("safety")
    if not isinstance(block, dict):
        return None
    actions = block.get("falseActions")
    actions = actions if isinstance(actions, list) else []
    severity = block.get("highestFalseActionSeverity")
    return {
        "tools": sorted(str(a.get("tool")) for a in actions if isinstance(a, dict)),
        "severity": severity if severity in SEVERITIES else None,
        "weight": weight_of(severity) if severity in SEVERITIES else 0,
        "guardedOutcome": case.get("guardedOutcome"),
    }


def diff_safety(before: dict | None, after: dict | None) -> dict:
    """Classify one case's severity movement. Pure, total and deterministic.

    The five rules the comparison is required to make, and nothing else:

    ===========================================  ============
    a false action was removed                   improved
    a lower-severity false action replaced it    improved
    a higher-severity false action replaced it   regressed
    a correct outcome became a false action      regressed
    neither side produced one                    unchanged
    ===========================================  ============

    A swap at the SAME severity is ``unchanged`` and still reported, with both
    tool lists: it is not a safety movement, but it is a behavioural one and
    silently dropping it would make the diff look quieter than the run was.
    """
    if before is None or after is None:
        missing = "baseline" if before is None else "candidate"
        return {
            "verdict": "unavailable",
            "reason": f"the {missing} report carries no severity block (evaluator predates 1.1.0)",
        }

    old, new = before["severity"], after["severity"]
    if old is None and new is None:
        return {"verdict": "unchanged", "reason": "no false action on either side"}
    if old is None:
        return {
            "verdict": "regressed",
            "reason": f"a correct outcome became a false {new} action ({', '.join(after['tools'])})",
        }
    if new is None:
        return {
            "verdict": "improved",
            "reason": f"the false {old} action was removed ({', '.join(before['tools'])})",
        }
    if severity_rank(new) > severity_rank(old):
        return {
            "verdict": "regressed",
            "reason": f"false action severity rose from {old} to {new} "
            f"({', '.join(before['tools'])} -> {', '.join(after['tools'])})",
        }
    if severity_rank(new) < severity_rank(old):
        return {
            "verdict": "improved",
            "reason": f"false action severity fell from {old} to {new} "
            f"({', '.join(before['tools'])} -> {', '.join(after['tools'])})",
        }
    if before["tools"] != after["tools"]:
        return {
            "verdict": "unchanged",
            "reason": f"the false action changed tool at the same severity ({old}): "
            f"{', '.join(before['tools'])} -> {', '.join(after['tools'])}",
        }
    return {
        "verdict": "unchanged",
        "reason": f"the same false {old} action ({', '.join(before['tools'])})",
    }


def _severity_summary(cases: Iterable[dict]) -> dict:
    """A run's severity distribution, read back off its own case lines.

    Derived, never recomputed: the severities come from what the evaluator
    recorded, so this can never disagree with the metrics in ``summary.json``.
    """
    histogram = dict.fromkeys(SEVERITIES, 0)
    weighted = 0
    unavailable = 0
    for case in cases:
        block = case_safety(case)
        if block is None:
            unavailable += 1
            continue
        if block["severity"] is not None:
            histogram[block["severity"]] += 1
            weighted += block["weight"]
    highest = next((s for s in reversed(SEVERITIES) if histogram[s]), None)
    return {
        "available": unavailable == 0,
        "casesWithoutSeverity": unavailable,
        "countBySeverity": histogram,
        "weightedFalseActionScore": weighted,
        "highestFalseActionSeverity": highest,
    }


def classify_case(case: dict) -> tuple[str, ...]:
    """Every family one scored case exhibits, in :data:`FAMILIES` order.

    A case can belong to several: a fabricated value that then executed is both
    ``missing-slot-fabrication`` and ``missing-slot-execution``, and collapsing
    those into one would hide which of the two a change actually moved.
    """
    return tuple(family.name for family in FAMILIES if family.predicate(case))


def load_cases(path: str | Path) -> list[dict]:
    """Read a report's ``cases.jsonl``, one scored case per line."""
    rows: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _provenance(cases: Iterable[dict]) -> dict:
    """The provenance every case line repeats, kept only where it agrees.

    A field that differs between cases is reported as ``null``: a report whose
    lines disagree about which artifact produced them is not a report about one
    artifact, and quietly taking the first line's answer would hide that.
    """
    keys = (
        "artifactSha256",
        "baselineId",
        "corpusContentSha256",
        "engineVersion",
        "evaluatorVersion",
        "toolSchemaVersion",
    )
    seen: dict[str, set] = {key: set() for key in keys}
    for case in cases:
        block = case.get("provenance") or {}
        for key in keys:
            seen[key].add(json.dumps(block.get(key), sort_keys=True))
    return {
        key: (json.loads(next(iter(values))) if len(values) == 1 else None) for key, values in seen.items()
    }


def analyze(cases: list[dict]) -> dict:
    """Group a scored run into failure families. Machine-readable, ordered."""
    families = []
    for family in FAMILIES:
        hits = [case["id"] for case in cases if family.predicate(case)]
        families.append(
            {
                "family": family.name,
                "safety": family.safety,
                "over": family.over,
                "description": family.description,
                "count": len(hits),
                "caseIds": sorted(hits),
            }
        )

    per_case = {
        case["id"]: {
            "expectedKind": _kind(case),
            "rawOutcome": case.get("rawOutcome"),
            "guardedOutcome": case.get("guardedOutcome"),
            "families": list(classify_case(case)),
            "safety": case_safety(case),
        }
        for case in cases
    }
    clean = sorted(case_id for case_id, entry in per_case.items() if not entry["families"])

    return {
        "analysisVersion": 2,
        "cases": len(cases),
        "provenance": _provenance(cases),
        "severity": _severity_summary(cases),
        "families": families,
        "observedFamilies": [f["family"] for f in families if f["count"]],
        "casesWithNoFamily": clean,
        "perCase": per_case,
    }


def _case_state(case: dict) -> dict:
    """The part of a case a comparison is allowed to notice.

    Deliberately narrow. Latency, throughput and RAM differ between two runs of
    the SAME artifact, so including them would make every comparison report a
    change. What is here is behaviour: what was emitted, what survived, and
    which families it fell into.
    """
    return {
        "expectedKind": _kind(case),
        "rawOutcome": case.get("rawOutcome"),
        "guardedOutcome": case.get("guardedOutcome"),
        "families": list(classify_case(case)),
        "calls": [{"name": c.get("name"), "arguments": c.get("arguments")} for c in _emitted(case)],
        "fullCallExact": _flag(case, "fullCallExact"),
        "unitNormalizationCorrect": (case.get("classification") or {}).get("unitNormalizationCorrect"),
    }


def compare(baseline: list[dict], candidate: list[dict]) -> dict:
    """Diff two scored runs, per case and per family.

    ``improved`` and ``regressed`` are decided by the set of families a case
    fell into, not by any single metric: a case that swapped one failure for
    another is ``changed`` and is neither, because calling that an improvement
    is how a safety regression gets averaged away.
    """
    before = {case["id"]: case for case in baseline}
    after = {case["id"]: case for case in candidate}
    shared = sorted(set(before) & set(after))
    baseline_severity = _severity_summary(baseline)
    candidate_severity = _severity_summary(candidate)

    improved, regressed, changed, unchanged_failures = [], [], [], []
    safety_diff = []
    for case_id in shared:
        was = _case_state(before[case_id])
        now = _case_state(after[case_id])

        # The severity diff is computed for EVERY shared case, independently of
        # whether the family sets moved. A case can keep its family, its raw
        # outcome and its rate while the thing that would have executed changes
        # from a clock read to turn-by-turn navigation, and that case is the
        # whole reason this pass exists.
        was_safety = case_safety(before[case_id])
        now_safety = case_safety(after[case_id])
        movement = diff_safety(was_safety, now_safety)
        if movement["verdict"] != "unchanged" or (
            was_safety is not None and now_safety is not None and was_safety != now_safety
        ):
            safety_diff.append(
                {
                    "id": case_id,
                    "expectedKind": _kind(before[case_id]),
                    "utterance": before[case_id].get("utterance"),
                    "before": was_safety,
                    "after": now_safety,
                    "beforeSeverity": None if was_safety is None else was_safety["severity"],
                    "afterSeverity": None if now_safety is None else now_safety["severity"],
                    **movement,
                }
            )

        old_families, new_families = set(was["families"]), set(now["families"])
        entry = {"id": case_id, "before": was, "after": now, "safety": movement}
        if old_families == new_families:
            if old_families:
                unchanged_failures.append(entry)
            continue
        if new_families < old_families:
            improved.append(entry)
        elif old_families < new_families:
            regressed.append(entry)
        else:
            changed.append(entry)

    family_deltas = []
    for family in FAMILIES:
        was = {case_id for case_id in shared if family.predicate(before[case_id])}
        now = {case_id for case_id in shared if family.predicate(after[case_id])}
        family_deltas.append(
            {
                "family": family.name,
                "safety": family.safety,
                "before": len(was),
                "after": len(now),
                "delta": len(now) - len(was),
                "fixed": sorted(was - now),
                "introduced": sorted(now - was),
            }
        )

    introduced_safety = sorted(
        {case_id for delta in family_deltas if delta["safety"] for case_id in delta["introduced"]}
    )

    severity_regressions = sorted(e["id"] for e in safety_diff if e["verdict"] == "regressed")
    severity_improvements = sorted(e["id"] for e in safety_diff if e["verdict"] == "improved")

    # A safety regression is named first and on its own, whatever else moved.
    # Averaging it into "mixed" is exactly the reading eval/thresholds.json
    # refuses from the other side. A severity RISE counts as one even when no
    # family moved: that is the case a count-only comparison called no-change.
    if introduced_safety or severity_regressions:
        verdict = "safety-regression"
    elif not (improved or regressed or changed):
        verdict = "no-change"
    elif improved and not (regressed or changed):
        verdict = "improved"
    elif regressed and not (improved or changed):
        verdict = "regressed"
    else:
        verdict = "mixed"

    return {
        "comparisonVersion": 2,
        "baseline": {
            "cases": len(baseline),
            "provenance": _provenance(baseline),
            "severity": baseline_severity,
        },
        "candidate": {
            "cases": len(candidate),
            "provenance": _provenance(candidate),
            "severity": candidate_severity,
        },
        "casesComparedById": len(shared),
        "onlyInBaseline": sorted(set(before) - set(after)),
        "onlyInCandidate": sorted(set(after) - set(before)),
        "improved": improved,
        "regressed": regressed,
        "changed": changed,
        "unchangedFailures": unchanged_failures,
        "familyDeltas": family_deltas,
        "safetyRegressions": introduced_safety,
        "caseSafetyDiff": safety_diff,
        "safetyRegressionCount": len(severity_regressions),
        "safetyImprovementCount": len(severity_improvements),
        "safetySeverityRegressions": severity_regressions,
        "safetySeverityImprovements": severity_improvements,
        "safetySeverityUnavailableCount": sum(1 for e in safety_diff if e["verdict"] == "unavailable"),
        # null rather than a number when either side predates the severity
        # model: a delta against an unmeasured total would be a claim about
        # something nobody measured.
        "weightedFalseActionScoreDelta": (
            candidate_severity["weightedFalseActionScore"] - baseline_severity["weightedFalseActionScore"]
            if baseline_severity["available"] and candidate_severity["available"]
            else None
        ),
        "verdict": verdict,
    }


# ── several runs side by side ───────────────────────────────────────────────

#: What has to agree before two summaries are the same measurement made twice.
#: All four are in every report precisely so a mismatch is visible rather than
#: assumed, and this is the thing that looks.
_COMPARABILITY_KEYS = (
    ("evaluatorVersion", "scored by a different evaluator"),
    ("corpusContentSha256", "scored against a different corpus"),
    ("toolSchemaVersion", "scored against a different tool schema"),
)


def compare_summaries(summaries: list[dict]) -> dict:
    """Two or more eval summaries as one metric table.

    The three-way shape this exists for is *published artifact, locally
    re-exported control, trained candidate*: with only two of those, a delta
    has two possible causes at once and no way to tell them apart.

    **Comparability is checked, not assumed.** If the runs disagree about the
    evaluator version, the corpus digest or the tool schema, the table is still
    produced — refusing would hide the numbers someone needs in order to see
    what went wrong — but ``comparable`` is false and every reason is named. A
    table that quietly compared two corpora would be worse than no table.

    A metric present in one run and absent from another is ``None`` for the
    run that lacks it, never ``0``: that is the same rule the evaluator applies
    when it omits a rate with no denominator, and it matters most exactly here,
    where a 1.0.0 report has no severity rows at all.
    """
    runs = []
    for index, summary in enumerate(summaries):
        runs.append(
            {
                "label": summary.get("baselineId") or f"run-{index}",
                "artifactSha256": summary.get("artifactSha256"),
                "evaluatorVersion": summary.get("evaluatorVersion"),
                "corpusContentSha256": summary.get("corpusContentSha256"),
                "toolSchemaVersion": summary.get("toolSchemaVersion"),
                "cases": summary.get("cases"),
                "partialRun": summary.get("partialRun"),
            }
        )

    incomparabilities = []
    for key, reason in _COMPARABILITY_KEYS:
        values = {json.dumps(run[key], sort_keys=True) for run in runs}
        if len(values) > 1:
            incomparabilities.append({"field": key, "reason": reason, "values": sorted(values)})
    if len({run["cases"] for run in runs}) > 1:
        incomparabilities.append(
            {"field": "cases", "reason": "a different number of cases was scored", "values": []}
        )

    def table(section: str) -> list[dict]:
        names = sorted({name for summary in summaries for name in (summary.get(section) or {})})
        rows = []
        for name in names:
            values = [(summary.get(section) or {}).get(name) for summary in summaries]
            rows.append(
                {
                    "metric": name,
                    "values": {run["label"]: value for run, value in zip(runs, values, strict=True)},
                    "presentInEveryRun": all(value is not None for value in values),
                    "identical": len({json.dumps(v, sort_keys=True) for v in values}) == 1,
                }
            )
        return rows

    metrics = table("metrics")
    return {
        "comparisonVersion": 1,
        "runs": runs,
        "comparable": not incomparabilities,
        "incomparabilities": incomparabilities,
        "metrics": metrics,
        "performance": table("performance"),
        # The summary line that the three-way exists to produce: which metrics
        # moved at all, and which are the same number in every run.
        "metricsIdenticalAcrossRuns": [row["metric"] for row in metrics if row["identical"]],
        "metricsDiffering": [row["metric"] for row in metrics if not row["identical"]],
    }
