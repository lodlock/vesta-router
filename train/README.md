# Training

**One model has been fine-tuned: `candidate-r1`. It is experimental, it is not
promoted, and its measured effect on the eval corpus is zero.**

`config/base.yaml` is still the *shape* a training config has to have, with
declared placeholders; `config/candidate-r1.yaml` is what was actually run. They
are separate files on purpose — overwriting base.yaml's placeholders with the
first thing anybody tried would turn "not yet chosen" into "chosen".

## The five steps, and where they stand

1. **Render** — `vesta_router.training.write_training_jsonl`, via
   `python -m vesta_router traindata --out <path>`. It takes the renderer, the
   system prompt and the tool JSON from the modules the *evaluator* uses, so the
   two cannot drift. **Done.**
2. **Fine-tune** — LoRA against the Needle 3 base, exporting a `.safetensors`
   adapter. No pickle, in either direction. **Done once** (r1).
3. **Export** — build the `.cact`, including quantization. **Done, and it is
   where r1's apparent result actually came from — see below.**
4. **Evaluate the `.cact`** — not the adapter. **Done**, three ways.
5. **Write the manifest** — from the build, not by hand. The *release* manifest
   is still unwritten because nothing has been promoted;
   `reports/candidate/<id>/build-manifest.json` is the record of one run and is
   deliberately not shaped like a release manifest.

## The exporter is a variable, and it was the dominant one

`candidate-r1` was compared against the **published** base `.cact` and appeared
to move eleven metrics. It did not. An untouched base checkpoint put through
candidate-r1's own exporter reproduces every one of them: control and candidate
produce identical values for all 23 metrics. The published archive is a
**mixed-width map** — 115 of its 119 body matmul tensors at CQ W2 — and the
local exporter writes W4 for all of them and cannot do otherwise. The first
comparison was between two quantizations.

Full account, including the byte-exact size decomposition and the three-way
table: [`../eval/EXPORT-CONTROL.md`](../eval/EXPORT-CONTROL.md).

```sh
python train/reexport_base.py \
  --cross-check artifacts/candidate-r1-adapter.safetensors \
  --record reports/analysis/local-base-reexport-build-record.json
```

`reexport_base.py` exists because **`needle build <checkpoint> --out x.cact`
does not export anything** — with no `--lora` and no depth change it copies the
published archive and says so, byte for byte. `--cross-check` rebuilds the same
artifact through the real `needle build` CLI with a zero-valued adapter and
asserts the two are identical, so "this is the candidate's own export path with
a null training delta" is demonstrated rather than claimed.

## Constraints that are not negotiable

- **Train only on `data/train`.** The fast CI tier checks disjointness from
  `data/eval` by id and by normalized utterance, and `config/base.yaml` sets
  `eval_dir: null` so an accidental edit is visible in review.
- **Exclude `data/historical`.** It is v1 material against a schema with
  different argument names; including it would teach `duration_seconds` to a
  model whose dispatcher expects `minutes`.
- **The seed is fixed** and lives in the config, so a build is reproducible from
  the revision recorded in its manifest.
- **The safety asymmetry belongs in the objective as well as the gates.** A
  refusal predicted as an action is the expensive direction; the reverse is
  merely unhelpful. `needle finetune` has no interface for it — r1 records that
  under `not_applied` rather than quietly dropping it.
- **A candidate is compared against the local re-export, never against the
  published archive.** New, and it is the rule r1 exists to have produced.

## What the first cycle answered, and what it did not

The open question was: *the spike found a 121M base fabricating required
arguments at confidence 1.00 — how much of that is the model's size and how much
was the four-tool toy schema?*

**Still open.** r1 cannot speak to it, because r1 changed nothing measurable: 57
steps at batch 8 over 151 examples, on CPU, final training loss 2.34, no
validation signal. "No effect" is an unsurprising outcome for that much training
and is not evidence about more of it.

What the cycle *did* answer is a question nobody had asked: **how much of a
locally built candidate's apparent delta belongs to its exporter.** For r1, all
of it.

Either way Vesta's grounding screen stays authoritative. A router that stops
fabricating is a better router, not a reason to remove the guard.
