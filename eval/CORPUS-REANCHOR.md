# Eval corpus digest: re-anchored to the committed LF bytes

**2026-09-23 · evaluator 1.1.0 · tool schema v2**

| | canonical digest of `data/eval` (`vesta_router.runner.corpus_digest`) |
|---|---|
| reports before this note | `c5b5577a29237794d53c184fddf1742c6fb406b88731680302a4cfe6271341ef` |
| **canonical from here on** | **`ed854cae5787de2edd623828fe3c0795fcbc8645b7b37e83b5250ebac1e4c020`** |

Same corpus, different bytes. The frozen eval corpus did not change; the digest
recorded for it was of a working tree git never stored.

## What happened

Every report up to this note was produced from a Windows checkout in which
`data/eval/positive.jsonl` had CRLF line endings — 21 lines, 21 extra bytes.
Git has only ever stored that file with LF (it was committed once, in
`415520f`). The other three eval files were LF in both places and hash
identically. `corpus_digest` hashes bytes, as it should, so the Windows tree
and a clean checkout produced different digests for the same cases.

The repository had no `.gitattributes`, so what a checkout contained depended
on the machine's `core.autocrlf`. It now has one:

```
*.jsonl text eol=lf
```

## Why this is a re-anchoring and not a corpus change

Evidence, all computed rather than asserted, in
[`reports/analysis/eval-corpus-lf-reanchor.json`](../reports/analysis/eval-corpus-lf-reanchor.json):

- **The difference is only line endings.** Converting the committed
  `positive.jsonl` to CRLF and re-running `corpus_digest` reproduces
  `c5b5577a…` exactly.
- **Scored content is identical.** For the published base, the local
  re-export and candidate-r1's historical reports, all 64 cases match the
  committed corpus on id, file and line order, utterance, tags, locale and
  expected kind / tool / arguments.

## What was re-run

The two controls every later candidate is compared with were re-evaluated on
the LF corpus, from WSL2:

| | artifact | LF-corpus report |
|---|---|---|
| local base re-export | `3bde318ede18…` | `reports/control/baseline-local-base-reexport-3bde318ede18-3bde318ede18-20260923T211627Z` |
| candidate-r1 | `8b62b28c1e28…` | `reports/candidate/baseline-candidate-r1-8b62b28c1e28-20260923T211657Z` |

Both artifacts were copied from the Windows tree with verified SHA-256, and
were also **rebuilt in WSL byte-identically** (`JAX_PLATFORMS=cpu`, the
cactus-needle 3.0.5 exporter). A GPU export of r1's adapter is *not*
byte-identical, so exports run on the CPU even when training does not.

Runtime: cactus-needle **3.0.1** — the wrapper from checkout `f189b23` via
`--needle-path`, and the 3.0.1 `libneedle.so` for linux x86_64 published in
`Cactus-Compute/needle3`. The historical reports used the same engine version
as a Windows `.dll`.

### The runtime platform is a variable too, and a small one

Against their historical Windows runs, both LF reports have **identical
aggregate metrics, identical tool choice on every case, and identical guarded
outcome on every case.** Seven of 128 case-runs differ in an *argument value*
the model chose for a call that was already wrong or already blocked — e.g.
`"He never texts me back."` → `navigate_to(destination="")` on Windows,
`navigate_to(destination="home")` on Linux. One of those moves a
fabrication/placeholder label on a negative case, which no slot metric counts.

Same engine version, different platform build: near-tied decodes can resolve
differently. That is why both controls were re-run rather than carried over,
and why **a candidate is compared only with controls evaluated on the same
runtime.** The case list is in the analysis record under
`historicalVsLfReports`.

## What is and is not edited

The historical reports are **not** rewritten. They are correct records of what
was measured, and their `corpusContentSha256` is true of the bytes they read.
Quote them for what they say; compare a new candidate against the LF-corpus
controls above.
