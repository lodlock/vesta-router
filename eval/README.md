# Evaluation

## Two tiers

**Fast tier** — every pull request. No GPU, no model, standard library only:
JSONL validation, id uniqueness, train/eval disjointness, tool-schema
validation, rendering determinism, manifest validation.

```sh
python -m vesta_router validate
python -m unittest discover -s tests
```

**Candidate tier** — on a candidate tag. Fine-tune, export the adapter, build
the final `.cact`, **evaluate that artifact**, run the full regression corpus,
emit `eval-report.json`, apply the gates.

The split is why a corpus fix does not wait on GPU time, and why a promotion
never happens without it.

**The harness that does the evaluating is built**, and it has been run against
the stock untrained `.cact`. [`BASELINE.md`](BASELINE.md) is the guide and holds
the first result; `reports/baseline/` holds the reports. The candidate tier is
still not wired into CI, because nothing trains yet.

## What the harness scores

Per case, and per tag, so a regression is localizable rather than a single
number that moved.

| Metric | |
|---|---|
| `toolSelectionAccuracy` | right tool, over cases that expect one |
| `argumentExactMatchAccuracy` | every argument exactly right, including units |
| `falsePositiveActionRate` | **safety** — an executable action for an utterance that asked for none |
| `missingSlotExecutionRate` | **safety** — an executable action with a required slot the user never gave |
| `noToolClassificationAccuracy` | chat correctly classified as chat |
| `unitNormalizationAccuracy` | `"two hours"` → `120` |
| `malformedOutputCount` | output the grammar was supposed to make well-formed and did not |
| `staleContextMisuseCount` | **safety** — a slot filled from a turn the router was not given |
| `fullCallExactMatchAccuracy` | right tool and right arguments, over every case expecting one |
| `falsePositiveToolCallRate` | a call for an utterance that asked for none, before the guard |
| `missingSlotFabricationRate` | the router invented a required value, emitted or withheld |
| `missingSlotEmissionRate` | it emitted the call rather than withholding it |
| `cancelNoActionRate` | an abandoned action produced nothing executable |
| `engineErrorCount` | the runtime failed |
| `stateLeakCount`, `nondeterministicCaseCount`, `resetFailureCount` | **safety** — whether the reset policy actually held |
| `medianLatencyMs`, `p95LatencyMs`, `minLatencyMs`, `maxLatencyMs` | when measured |
| `medianPrefillTps`, `medianDecodeTps` | when the runtime reports them |
| `peakMemoryMb` | when measured |

**Absent is a value.** A metric the harness did not produce is omitted from the
report, never reported as `0` — on a false-positive-rate row, `0` reads as
perfect. The manifest validator enforces the same thing, and a gate whose metric
is absent **fails**.

`peakMemoryMb` in a baseline report is the **engine's own** per-call reading on
the host, reported as the maximum over the run. It is not a device figure and
not a process measurement. The spike's ~131 MB is a reading from one phone and
stays where it was recorded; device benchmarking becomes part of promotion when
there is a harness to produce it, and it is not a gate for getting training
started.

## Scoring `incomplete`

A case expecting `incomplete` passes when **no executable action reaches
dispatch**. The router may well propose the tool with a fabricated argument;
Vesta's grounding screen refuses it and the user is asked. That is the system
behaving correctly.

The proposal is still counted, separately, because a router that leans on the
guard every time will be wrong the day the guard has a gap. Two numbers, two
questions, and collapsing them would hide the second.

## Thresholds

`thresholds.json`. Every one is **provisional** until a baseline model exists,
except `malformedOutputCount` — Needle's grammar claims well-formed output is
guaranteed by construction, so a non-zero value is a finding about the engine
rather than a quality score, and it should stop a release until it is
understood.

Safety gates are not traded against accuracy. Perfect tool-selection accuracy
does not unblock a single missing-slot execution, and a test asserts that.

```sh
python -m vesta_router gate examples/eval-report-1.0.0.json
```

## The harness

```sh
python -m vesta_router eval --model /path/to/model.cact
```

It lives in the package rather than in an `eval/harness/` of its own —
`vesta_router.evaluate` (classification, scoring, aggregation),
`vesta_router.grounding` (the evaluator-side screen),
`vesta_router.runner` (passes, reset policy, report) and
`vesta_router.engines` (the runtime boundary). One import root, and every part
of it that can be wrong without a model is unit-tested without one, against
recorded envelopes.

**`vesta_router.engines.needle` is the only module in this repository that
imports a third-party runtime**, and it does so inside the function that needs
it. `pyproject.toml` still declares no dependencies and the fast tier still runs
on the standard library alone.

It loads the `.cact` through Needle's own engine, renders each case through
`vesta_router.render`, resets between every case, screens the result, and emits
`summary.json` / `cases.jsonl` / `manifest.json`. What it will not do is
produce a number any other way: if the artifact cannot be loaded the run stops
with a non-zero exit and no report, because a report produced without the
artifact is a number about something else.

[`BASELINE.md`](BASELINE.md) covers all of it — reset policy, raw versus
post-guardrail metrics, report format, the limits of a host-side run, and how a
trained candidate is compared against the baseline.
