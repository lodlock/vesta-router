# Training

**Nothing here trains a model yet, and there is no script that pretends to.**

`config/base.yaml` exists because its git revision is recorded in every manifest
it will eventually produce (`provenance.trainingConfigRevision`), so the shape
had to be settled before the first build rather than after it. Every
hyperparameter in it is a placeholder, marked as such.

## What has to exist

1. **Render** — each training example through `vesta_router.render`, which is
   already written and tested. Vesta's future router caller must render
   byte-identically, or the model sees a prompt it was never trained on.
2. **Fine-tune** — LoRA against the Needle 3 base, exporting a
   `.safetensors` adapter. No pickle, in either direction.
3. **Export** — build the `.cact`, including quantization.
4. **Evaluate the `.cact`** — not the adapter. Quantization changes behaviour,
   and a number measured before it is a number about a different file.
5. **Write the manifest** — from the build, not by hand, with the artifact's
   real digest and the three revisions it was built from.

Steps 1 and 5 are done (`src/vesta_router/`). Steps 2–4 need the Needle training
toolchain.

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
  merely unhelpful.

## The open question this is meant to answer

The spike found a 121M base model fabricating required arguments at confidence
1.00 on under-specified prompts. **How much of that is the model's size and how
much was the four-tool toy schema is unknown**, and it is not assumed to be
fixable by fine-tuning. The first train/eval cycle answers it; if the answer is
"the size", the corpus and the harness are still exactly what a larger base
needs.

Either way Vesta's grounding screen stays authoritative. A router that stops
fabricating is a better router, not a reason to remove the guard.
