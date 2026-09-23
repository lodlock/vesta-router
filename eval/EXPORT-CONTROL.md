# The export control, and what it did to candidate-r1's result

**2026-09-23 · evaluator 1.1.0 · tool schema v2 · corpus `c5b5577a29237794`**

candidate-r1 was measured against the **published** base `.cact` and reported a
set of movements — argument accuracy up, tool selection down, one fewer stale
context. This document is the run that says how much of that was the training.

**The answer is none of it.**

An untouched base checkpoint, put through candidate-r1's own exporter with no
adapter, reproduces **every** one of those movements exactly. On the same 64
cases, with the same evaluator and the same runtime, the untrained control and
the trained candidate produce **identical values for all 23 metrics**.

---

## Why a control was needed at all

candidate-r1's own build record flagged it before this run existed:

> 63.4 MB against the published base archive's 35.3 MB for the same 20-layer
> architecture at W4A8. […] whether the published archive packs something this
> one does not is UNEXPLAINED

and

> It is the PUBLISHED base `.cact`, not a local export of `baseCheckpoint`, so
> some part of any delta may belong to the exporter rather than to training.

Two artifacts that differ in two ways cannot attribute a delta to either. The
third artifact is what makes the question answerable:

| | what it is | what it isolates |
|---|---|---|
| **A** published base `.cact` | Cactus's own build, shipped on HuggingFace | — |
| **B** local base re-export | the same checkpoint, **candidate-r1's exporter**, no adapter | A → B is the **build path** |
| **C** candidate-r1 `.cact` | the same checkpoint, the same exporter, **with the LoRA merged** | B → C is the **training** |

---

## The obvious re-export command does not re-export

The first thing to try is:

```sh
needle build checkpoints/needle3.safetensors --out local-base.cact
```

It produces a file byte-identical to the published archive — sha256
`c9d915eca282…`, all 35,335,380 bytes of it — because it never reads the
checkpoint. `needle.model.finetune.build_main` short-circuits:

```python
if not args.lora and (not layers or layers == read_layers(base_archive)):
    shutil.copyfile(base_archive, out)
    print("... the published base archive")
```

It even says so in its own output. **The obvious control reproduces the file it
was supposed to control for**, and would have looked like a successful null
result while measuring nothing. `--layers 20` does not help either: the guard
treats a depth equal to the base's as no depth change.

`train/reexport_base.py` takes the export branch directly instead —
`load_checkpoint` → drop the confidence head → `rung(…, None)` →
`write_export(bits=WEIGHT_BITS, tokenizer=<the published archive's blob>,
kv_window=effective_kv_window(config))` — which is exactly what `build_main`
runs on the branch candidate-r1 took, minus the LoRA merge.

### And it proves that rather than asserting it

`--cross-check` builds the same artifact a second time through the **real
`needle build` CLI** with a zero-valued adapter: candidate-r1's exact command,
its exact LoRA key set, rank 16 and scale 2.0, with every `B` set to zero so
the merged delta is exactly `scale * (A @ 0) == 0`.

```
crosscheck needle build checkpoints/needle3.safetensors \
             --lora artifacts/local-base-zero-adapter.safetensors --out …
           cli sha256 3bde318ede183eb0…  IDENTICAL
```

The two files are byte-identical. So the no-adapter export **is** what
candidate-r1's command produces on a null training delta, and the control is
exact rather than argued. Two independent runs also produced the same digest,
so the export is reproducible.

---

## The artifacts

| | published base | local re-export | candidate-r1 |
|---|---|---|---|
| sha256 | `c9d915eca282ed42…` | `3bde318ede183eb0…` | `8b62b28c1e281e4b…` |
| bytes | 35,335,380 | **63,437,076** | **63,437,076** |
| tensors | 581 | 574 | 574 |
| CQ weight widths | **115 × W2, 4 × W4** | **119 × W4** | **119 × W4** |
| probe heads | confidence (7 tensors) | none | none |
| tokenizer blob | identical in all three | | |
| geometry | identical in all three — 20 layers, d_model 768, vocab 8192, kv_bits 8 | | |

The re-export is the same size as the candidate to the byte. That is the first
sign of what the size discrepancy actually was.

---

## Where the 28 MB went — every byte of it

Read off the tensor directory by `python -m vesta_router inspect A B --compare`.
Each record carries a dtype, a shape, a byte count, a quantization group **and a
bit width**, so this is not an estimate:

```
totalDeltaBytes        28101696
  modelBodyTensorBytes 28139520   115 matmul tensors, CQ W2 -> CQ W4
  probeHeadBytes         -37276   the confidence head, dropped by the local build
  tensorDirectoryBytes     -308   7 fewer records
  alignmentPaddingBytes    -240   64-byte alignment over 7 fewer tensors
  codebookBytes               0
  unexplainedBytes            0
```

**The cause is weight precision, and nothing else.** It is not LoRA — the
untrained control is the same size. It is not compression, packing, duplicated
tokenizer data or duplicated metadata: the tokenizer blob is byte-identical in
all three and the codebook is unchanged.

The published archive is a **mixed-width deployment map**: 115 of its 119 body
matmul tensors are stored at CQ **W2**, with only the embedding and three mHC
blocks at W4. The local exporter writes **W4 for every one of them**, and
cannot do otherwise — `WEIGHT_BITS = 4` is a module constant in
`needle.model.quantize` and `needle.model.export._cq_pack` *raises* on any other
width:

```python
if bits != WEIGHT_BITS:
    raise ValueError(f"CQ packing supports bits={WEIGHT_BITS}; got bits={bits}")
```

The remaining 7-tensor difference is the **confidence head** — a
`heads.manifest` carrying one code plus that head's six tensors. `needle build`
drops it whenever a LoRA is merged, because local tuning does not train it, and
`train/reexport_base.py` drops it too so the control matches the candidate's
tensor set exactly.

### What that means, stated carefully

- **The published base and every locally built artifact are different
  quantizations of the same weights.** "Nominally the same architecture and
  quantization" was the assumption this run was built to test, and it was wrong.
  Comparing a W4 candidate against a W2 baseline was comparing two things.
- **The open-source toolchain cannot reproduce the published archive.** There is
  no per-tensor width flag, no deployment map and no `--bits`. Whatever produced
  the mixed map is not in this package. That is a bound on what any local
  experiment can conclude about the shipped model, and it does not go away by
  training more.
- **Nothing here says which is better.** W4 is more precise per weight and 1.8×
  the size; this run measures behaviour, not quality, and the behaviour is worse
  on tool selection and better on arguments. Which numerics Vesta should ship is
  a separate decision that needs a device-side measurement.

---

## The three-way result

Same evaluator (1.1.0), same 64-case corpus (content digest
`c5b5577a29237794…`), same tool schema (v2), same prompt settings, same runtime
(`cactus-needle` 3.0.1), same host, reset between every case, isolation pass on.

`python -m vesta_router compare-runs <three summary.json> --output …`

| metric | **A** published base | **B** local re-export | **C** candidate-r1 | B → C |
|---|---|---|---|---|
| `toolSelectionAccuracy` | 0.964286 | 0.928571 | 0.928571 | — |
| `argumentExactMatchAccuracy` | 0.518519 | 0.692308 | 0.692308 | — |
| `fullCallExactMatchAccuracy` | 0.500000 | 0.642857 | 0.642857 | — |
| `noToolClassificationAccuracy` | 0.315789 | 0.263158 | 0.263158 | — |
| `falsePositiveToolCallRate` | 0.684211 | 0.736842 | 0.736842 | — |
| `falsePositiveActionRate` | 0.526316 | 0.526316 | 0.526316 | — |
| `missingSlotFabricationRate` | 0.428571 | 0.500000 | 0.500000 | — |
| `missingSlotPlaceholderRate` | 0.285714 | 0.285714 | 0.285714 | — |
| `missingSlotEmissionRate` | 0.857143 | 0.928571 | 0.928571 | — |
| `missingSlotExecutionRate` | 0.214286 | 0.214286 | 0.214286 | — |
| `cancelNoActionRate` | 0.666667 | 0.666667 | 0.666667 | — |
| `unitNormalizationAccuracy` | 0.400000 | 0.400000 | 0.400000 | — |
| `staleContextMisuseCount` | 5 | 4 | 4 | — |
| `malformedOutputCount` | 0 | 0 | 0 | — |
| `engineErrorCount` | 0 | 0 | 0 | — |
| `stateLeakCount` | 0 | 0 | 0 | — |
| `nondeterministicCaseCount` | 0 | 0 | 0 | — |
| `resetFailureCount` | 0 | 0 | 0 | — |
| `weightedFalseActionScore` | 167 | **193** | **193** | — |
| `highestFalseActionSeverity` | high_impact | high_impact | high_impact | — |
| `falseActionCountBySeverity` | 6/2/3/1/2 | 7/2/3/0/3 | 7/2/3/0/3 | — |
| `falsePositiveActionCountBySeverity` | 6/1/1/1/1 | 6/1/1/0/2 | 6/1/1/0/2 | — |
| `falsePositiveActionRateBySeverity` | see report | see report | see report | — |

*(severity histograms read `read_only / reversible_low / user_visible_side_effect
/ external_side_effect / high_impact`.)*

**Twelve metrics move from A to B. Zero move from B to C.**

Latency and memory, for completeness — host figures, never device figures:

| | A | B | C |
|---|---|---|---|
| median latency | 213.98 ms | 218.27 ms | 217.35 ms |
| p95 latency | 310.60 ms | 303.55 ms | 295.11 ms |
| median prefill | 469.15 tok/s | 433.90 tok/s | 431.35 tok/s |
| median decode | 259.40 tok/s | 258.15 tok/s | 260.15 tok/s |
| engine peak RSS | **120.3 MB** | **164.8 MB** | **164.7 MB** |

The +44 MB of resident memory is the W2 → W4 map again, and it is the one
practical consequence of the export difference that a phone would notice.

### Per case

```
python -m vesta_router regressions <B>/cases.jsonl <C>/cases.jsonl
  improved 0  regressed 0  changed 0  unchanged failures 39
  verdict  no-change
```

Zero failure families moved, zero severities moved, `weightedFalseActionScoreDelta` 0.

The candidate is not *bit-identical* to the control, and the difference is worth
naming precisely because it is so small. Ten of 64 cases differ at all. Eight
differ only in the wording of the engine's `reasoning` text. **Two differ in
what they emitted:**

| case | control | candidate-r1 |
|---|---|---|
| `slot.alarm.bare.it` — *"Mettimi una sveglia."* | `set_alarm {"time": "07:00"}` | `set_alarm {"time": "now"}` |
| `stale.cancellation.it` — *"lascia stare"* | `set_alarm {"label": "Pending", "time": "missing"}` | `set_alarm {"time": "07:00"}` |

Both are fabrications on either side, both are refused by the grounding screen
on either side, and both land in the same failure families with the same
severity. **Three epochs over 151 examples changed the shape of two fabricated
values and moved no metric.**

The published-base comparison is the other half of the finding: `A → B` and
`A → C` produce the *same* diff — 7 improved, 3 regressed, 4 changed — and
differ from each other in exactly one field, the fabricated value on
`slot.alarm.bare.it`.

---

## What this does not say

1. **Not that fine-tuning cannot work here.** It says this run did not move this
   corpus. 57 steps at batch 8 over 151 examples, on CPU, with a final training
   loss of 2.34 and no validation signal, is a very small amount of training;
   "no effect" is an unsurprising outcome for it and is not evidence about a
   larger one.
2. **Not that the published model is better.** Twelve metrics differ and they
   differ in both directions. Which numerics to ship is a device-side question.
3. **Not that the corpus is insensitive.** It separated A from B cleanly, on
   twelve metrics and eleven cases.
4. **Not that candidate-r1 is safe to promote.** It is not promoted, it is not
   stable, and it is not a release. Its *only* demonstrated property is that it
   behaves like its own untrained control.

---

## Reproducing it

```sh
# 1. the control artifact (~30 s on CPU; downloads the published base archive)
python train/reexport_base.py \
  --cross-check artifacts/candidate-r1-adapter.safetensors \
  --record reports/analysis/local-base-reexport-build-record.json

# 2. what is actually in each archive
python -m vesta_router inspect <published>.cact artifacts/local-base-reexport-*.cact --compare --json

# 3. three evaluations, one evaluator
python -m vesta_router eval --model <published>.cact                        --output reports/baseline
python -m vesta_router eval --model artifacts/local-base-reexport-*.cact    --output reports/control
python -m vesta_router eval --model artifacts/candidate-r1.cact             --output reports/candidate

# 4. the table, and the per-case diffs
python -m vesta_router compare-runs <A>/summary.json <B>/summary.json <C>/summary.json --output …
python -m vesta_router regressions  <B>/cases.jsonl  <C>/cases.jsonl --output …
```

`compare-runs` checks comparability rather than assuming it: a mismatched
evaluator version, corpus digest, tool schema or case count sets
`comparable: false` and names the field. The table is still produced — the
numbers are what a reader needs in order to see what went wrong — but it never
claims to be a comparison it is not.

---

## Before a second training run

**The exporter confound is understood and bounded, and that was the blocker.**
A local build's numerics are now a known quantity: uniform W4A8, no confidence
head, 574 tensors, reproducible, and provably the same file the candidate's own
command produces on a null delta.

So r2 is unblocked **on this axis**, with one standing rule and one open
question:

- **The rule: a candidate is compared against the local re-export, never against
  the published archive.** The published archive is a different quantization.
  Comparing against it attributes a precision change to training, which is
  precisely the error this run found.
- **The open question, which is not a blocker: what Vesta would actually ship.**
  If the answer is the published W2 map, then no locally trained artifact is
  the shipped artifact, and every number here describes a model with 1.8× the
  weight bytes and +44 MB of RSS. That is a question for the platform
  fine-tuning path (`needle platform finetune`), which keeps the confidence head
  and presumably the deployment map, and it should be answered before a
  candidate is proposed for promotion — not before r2 is run.

What r2 should change first is the **training**, not the export: r1's null
result is about 57 steps, not about the corpus or the harness.
