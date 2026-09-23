# Reports

Evaluation runs, one directory each.

```
baseline/<baseline-id>-<UTC timestamp>/      an untrained reference artifact
control/<baseline-id>-<UTC timestamp>/       an untrained artifact rebuilt LOCALLY
candidate/<baseline-id>-<UTC timestamp>/     a trained candidate, same harness
  summary.json    metrics, performance, per-tag breakdown, outcome counts, provenance
  cases.jsonl     one line per case: prompt, raw envelope, verdicts, classifications, severity
  manifest.json   artifact, runtime, prompt, tool schema, corpus digest, severity model, policy, host

analysis/
  <id>-failures.json           one run grouped by failure family
  <before>-vs-<after>.json     two runs diffed, per case and per family
  three-way-*.json             N runs in one metric table
  local-base-reexport-build-record.json   how the control artifact was built
```

`baseline/`, `control/` and `candidate/` are the same report, written by the
same code, and the directory says only which side of a comparison it was
produced for — the harness's own identifier is `baseline-<artifact stem>-<sha12>`
in all three, because it names a measurement of one file and not a status.

## Why `control/` exists

A candidate and a published baseline differ in **two** ways at once: the
training, and whatever the two build paths do differently. `control/` holds the
third artifact that separates them — the *same checkpoint* the candidate was
trained from, put through the *candidate's own exporter*, with no adapter.

It is not an optional nicety. candidate-r1's entire apparent improvement over
the published base is reproduced by that untrained control, because the
published archive is quantized differently from anything this toolchain can
build. See [`../eval/EXPORT-CONTROL.md`](../eval/EXPORT-CONTROL.md).

Its identifier reads `baseline-local-base-reexport-<sha12>-<sha12>`. The
repetition is real and harmless: the artifact's *filename* carries the digest
(so two control builds can sit side by side) and the harness then derives
`baseline-<stem>-<sha12>` from it, as it does for every artifact.

## The derived documents

The files under `analysis/` are derived, not measured: they are `cases.jsonl`
and `summary.json` re-read by `vesta_router.failures`, so they can be
regenerated from a report at any time and carry no number the report does not.

```sh
python -m vesta_router failures    reports/<...>/cases.jsonl --output reports/analysis/<id>-failures.json
python -m vesta_router regressions  reports/<a>/cases.jsonl reports/<b>/cases.jsonl --output reports/analysis/<a>-vs-<b>.json
python -m vesta_router compare-runs reports/<a>/summary.json reports/<b>/summary.json reports/<c>/summary.json --output reports/analysis/three-way-<...>.json
```

`compare-runs` checks comparability rather than assuming it: a mismatched
evaluator version, corpus digest, tool schema or case count sets
`comparable: false` and names the field. It still prints the table — those
numbers are what a reader needs in order to see what went wrong — but it never
claims to be a comparison it is not.

**These are committed, and the weights they describe are not.** A report is a
few hundred kilobytes, it is the part that has to be auditable, and it outlives
the artifact it measures — the same reason `releases/` carries manifests and
eval reports rather than `.cact` files.

## Evaluator versions on disk

A report carries the `evaluatorVersion` that wrote it, and two reports carrying
different ones are not comparable. Both are here, and both are kept:

| | |
|---|---|
| **1.0.0** | the first baseline (`…034052Z`) and the first candidate run (`…152523Z`). No severity block, so a diff against one reports severities as `unavailable` — never as absent failures. |
| **1.1.0** | the three runs timestamped `…1742`–`…1744`. **These are the ones to quote.** |

Three directories carry a `SUPERSEDED.md`. They were produced mid-change by an
evaluator that *called* itself 1.1.0 before the severity denominator was
widened, so they report the same version with different severity numbers —
exactly the trap the version field exists to prevent. They are kept rather than
deleted, because a deleted report is one nobody can check, and each one says
what is wrong with it. Their unweighted metrics are unaffected.

A report here is a **benchmark result, never a promotion**. `baseline-…` names a
measurement of one file by its digest; it is not a `ModelVersion` and
`parse_model_version` returns `None` for it. Promotion is a separate decision,
made against `eval/thresholds.json`, and it requires a manifest.

How to produce one, and what every number in it means:
[`../eval/BASELINE.md`](../eval/BASELINE.md). What the severity rows mean:
[`../eval/SEVERITY.md`](../eval/SEVERITY.md).

```sh
python -m vesta_router eval --model /path/to/model.cact
python -m vesta_router gate reports/baseline/<dir>/summary.json
```

`releases/<version>/` is the other half of this: the record of a *promoted*
build. **Nothing has been promoted.** One model has been trained
(`candidate-r1`); it is experimental, its measured effect on this corpus is
zero, and it is not a promotion candidate.
