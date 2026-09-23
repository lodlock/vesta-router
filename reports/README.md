# Reports

Evaluation runs, one directory each.

```
baseline/<baseline-id>-<UTC timestamp>/
  summary.json    metrics, performance, per-tag breakdown, outcome counts, provenance
  cases.jsonl     one line per case: prompt, raw envelope, verdicts, classifications
  manifest.json   artifact, runtime, prompt, tool schema, corpus digest, policy, host
```

**These are committed, and the weights they describe are not.** A report is a
few hundred kilobytes, it is the part that has to be auditable, and it outlives
the artifact it measures — the same reason `releases/` carries manifests and
eval reports rather than `.cact` files.

A report here is a **benchmark result, never a promotion**. `baseline-…` names a
measurement of one file by its digest; it is not a `ModelVersion` and
`parse_model_version` returns `None` for it. Promotion is a separate decision,
made against `eval/thresholds.json`, and it requires a manifest.

How to produce one, and what every number in it means:
[`../eval/BASELINE.md`](../eval/BASELINE.md).

```sh
python -m vesta_router eval --model /path/to/model.cact
python -m vesta_router gate reports/baseline/<dir>/summary.json
```

`releases/<version>/` is the other half of this: the record of a *promoted*
build. Nothing is promoted yet, and nothing here has been trained.
