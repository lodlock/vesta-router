# Corpora

JSONL, one case per line, no wrapper array. Three directories, and the
separation between the first two is structural rather than a convention somebody
remembers.

| | |
|---|---|
| `train/` | what the model learns from |
| `eval/` | the regression corpus. **Never trained on.** |
| `historical/` | frozen v1 spike material. Not trained on, not evaluated against v2. |

Validate everything:

```sh
python -m vesta_router validate
```

## Disjointness

`train/` and `eval/` must not overlap **by id or by normalized utterance**.
Normalization is casefolded, accent-folded, punctuation-stripped and
whitespace-collapsed, which is deliberately aggressive: two cases that differ by
a comma are the same sentence for the purpose of asking whether the model was
trained on the thing it is being measured on.

Checking ids alone would not be enough. Ids are easy to keep distinct and prove
nothing; the same sentence under two names is the actual memorization risk.

## Fields

Every case: `id` (unique across the whole repository, and stable — results are
cited by it), `utterance` (exactly what the user said, typos and STT artefacts
kept), `locale` (`en` or `it`), and `state` (what Vesta supplied for this
decision; `null` means none).

Eval cases add `expect`, `tags` and `source`. Training cases add `output` and
`category`. Two names for one shape, kept apart on purpose: an eval case states
what must happen and a training case states what to learn, and a file that
confuses them is a file in the wrong directory.

`source` is `spike-device-2026-09-22` for a case that records an observed device
result, `authored` otherwise. Only the first kind is evidence.

## Outcomes

`tool`, `incomplete`, `cancel`, `chat`.

**`incomplete` asserts an outcome, not an output**: no executable action may
reach dispatch. The router proposing a tool with a fabricated required argument
and Vesta's grounding screen refusing it is a pass — the user gets asked. How
often the router fabricates is scored separately.

## Tags

Free-form, and the harness reports per tag so a regression is localizable.
Load-bearing ones:

| Tag | |
|---|---|
| `safety` | a case where getting it wrong costs the user something real |
| `high-cost` | `make_call` and `send_sms` — irreversible by the time anyone notices |
| `missing-slot`, `stale-context` | the two zero-tolerance gates |
| `unit-conversion` | minutes from seconds, hours, halves, quarters and compounds — the closed set `set_timer` has to get exactly right |
| `trap`, `near-miss` | an utterance that contains the tool's whole vocabulary and is not a request |
| `stt`, `typo` | noisy input, which the assist surface produces routinely |
| `italian` | asserted present in both corpora |

The `unit-conversion` set is deliberately small and complete: `20 minutes` → 20,
`30 seconds` → 0.5, `90 seconds` → 1.5, `two hours` → 120, `half an hour` → 30,
`1 minute 30 seconds` → 1.5. Accepted precision is a whole second, read off the
production dispatcher rather than chosen — `system-actions.ts` refuses
`minutes <= 0` and computes `Math.round(minutes * 60)`. The first baseline got
two of five right, missing every downward conversion, so the signal these cases
carry is real.

## Italian is not optional

Vesta is bilingual from day one, so the router is too. A test asserts both
locales are present in both corpora: a corpus that drifts to English-only
produces a router that is English-only, and nothing else would notice.

## `historical/`

The v1 corpus, recorded against the spike's four toy tools. Kept so a spike
result can still be read in the terms it was produced in — `set_timer` took
`duration_seconds` there, and `control_light` and `open_app` exist in no
dispatcher.

Its ids are prefixed `v1.`, because the v2 corpus reuses the same naming
convention and a cited result must never be ambiguous about which schema it was
measured under. Tests assert it still validates against v1, that it does **not**
validate against v2, and that none of its ids leak into the live corpora.
