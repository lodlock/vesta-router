# Superseded — do not quote this run

This run was produced mid-change by an evaluator that called itself **1.1.0**
and was not the 1.1.0 that shipped. Between it and the final runs,
`weightedFalseActionScore`, `falseActionCountBySeverity` and
`highestFalseActionSeverity` were widened from "cases expecting no action"
(`chat`, `incomplete`, `cancel`) to **every case**, so that a false action on a
case expecting a *different* tool is counted — and so that the number agrees
with the one `vesta_router.failures` computes from the same `cases.jsonl`.

So two runs on this disk carry the same `evaluatorVersion` and different
severity numbers, which is exactly the trap the version field exists to
prevent. It is kept rather than deleted because everything else in it is real,
and because a deleted report is one nobody can check.

**Its unweighted metrics are unaffected** and match the final run exactly —
the change touched only the severity rows.

The runs to quote are the ones timestamped `20260923T1742`–`20260923T1744`:

| | |
|---|---|
| published base | `reports/baseline/baseline-needle3-c9d915eca282-20260923T174256Z` |
| local re-export | `reports/control/baseline-local-base-reexport-3bde318ede18-3bde318ede18-20260923T174345Z` |
| candidate-r1 | `reports/candidate/baseline-candidate-r1-8b62b28c1e28-20260923T174434Z` |

See `eval/EXPORT-CONTROL.md`.
