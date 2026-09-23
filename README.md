# vesta-router
Training, evaluation, and release pipeline for Vesta’s local Needle-based tool router.

**Status: baseline evaluation complete. One experimental fine-tune has been run.
No model is promoted for production use.**

What exists is the contract, the corpora, the part of the pipeline that can be
right or wrong without a GPU — corpus validation, the state-block renderer,
manifest validation and the promotion gates — **the evaluation harness with a
measured baseline of the stock untrained `.cact`**, and a false-action
**severity model** so a report can tell a wasted turn from a phone call staged
to a named contact. All of the first part runs on the standard library:

```sh
python -m vesta_router validate      # corpora, tool schema, disjointness, rendering
python -m unittest discover -s tests
```

The harness is the one thing that loads a runtime, and it is the one thing that
is not run in pull-request CI:

```sh
python -m vesta_router eval --model /path/to/model.cact
```

[`eval/BASELINE.md`](eval/BASELINE.md) is the guide and holds the first result.
Short version: against tool schema v2, stock Needle 3 picks the right tool
**0.964** of the time and has almost no idea when *not* to act —
`noToolClassificationAccuracy` **0.316**, and **0.526** of the no-tool corpus
produced an action a deterministic guard would have let through. Zero malformed
outputs, zero state leaks. All three safety gates fail, which is the expected
shape for an untrained model and the reason to measure it before changing
anything.

### candidate-r1 is experimental, and its result is a negative one

`candidate-r1` is the first fine-tune: LoRA rank 16, three epochs over 151
examples, 57 steps on CPU. It is **not stable, not promoted and not a release**,
and it is not a candidate for becoming one.

Measured against the *published* base archive it appeared to move eleven
metrics. It did not. An **untrained** base checkpoint put through candidate-r1's
own exporter, with no adapter, reproduces every one of those movements exactly:
on the same 64 cases the untrained control and the trained candidate produce
**identical values for all 23 metrics**, zero failure families move and zero
severities move. Three epochs changed the shape of two fabricated values and
nothing else.

The confound was the artifact, not the corpus. The published archive stores 115
of its 119 body matmul tensors at **CQ W2**; the local exporter writes **W4** for
all of them and cannot do otherwise. So the original comparison was between two
quantizations — it also accounts for 63.4 MB against 35.3 MB, and for +44 MB of
resident memory, to the byte and with no residual.

[`eval/EXPORT-CONTROL.md`](eval/EXPORT-CONTROL.md) is that run: how the control
was built and cross-checked, where every byte of the size difference went, the
three-way metric table, and what has to hold before a second training run.
[`eval/SEVERITY.md`](eval/SEVERITY.md) is the severity model it is scored with.

[Vesta](https://github.com/lodlock/vesta) is a local-first AI assistant that runs
entirely on-device. This repository holds the small model that decides what a
user's message *is* — an action, a question, a chat — so the app does not have to
load a multi-billion-parameter model to set a timer.

---

## The one rule everything else follows

> **The router proposes. Vesta decides.**

The model produces an intent, a tool and arguments. Vesta owns deterministic
grounding, missing-slot enforcement, unit and format normalization, compatibility
checks, execution authorization, and what conversation state the router is even
allowed to see.

**Model confidence is never execution permission.** On device, for the utterance
*"Wake me up."* — which contains no time — Needle proposed `time: "07:00"`,
emitted it, and reported **confidence 1.00**. Nothing in this repository or in
Vesta's router-model code accepts a confidence value; there is nowhere to pass
one, and that is on purpose.

The full contract is in the Vesta repository at
[`docs/router-model-contract.md`](https://github.com/lodlock/vesta/blob/main/docs/router-model-contract.md).
[`CONTRACT.md`](CONTRACT.md) here records which side owns what.

---

## Layout

```
data/
  train/        training corpus, JSONL, 151 cases
  eval/         regression corpus, JSONL, 64 cases - never trained on
  historical/   frozen v1 spike material, validated against the v1 schema
tools/
  tool-schema-v1.json   FROZEN. The spike's four toy tools.
  tool-schema-v2.json   ACTIVE. Derived from Vesta's production tool registry.
train/
  config/       training configuration, revisioned; its revision goes in every manifest
  reexport_base.py      the untouched-base export control - see eval/EXPORT-CONTROL.md
eval/
  thresholds.json       promotion gates. Every threshold is PROVISIONAL.
  BASELINE.md           how to run the evaluator, and the first baseline result
  EXPORT-CONTROL.md     published vs locally exported base, and candidate-r1's real delta
  SEVERITY.md           the false-action severity taxonomy and its metrics
examples/       worked manifest, eval report and update index
releases/       <version>/manifest.json and eval-report.json per promoted build
reports/        baseline|control|candidate/<id>-<timestamp>/{summary,cases,manifest}
                analysis/  derived failure groupings and run diffs
src/vesta_router/   validation, rendering, manifest and gate logic, the
                    evaluator (evaluate, grounding, runner, engines/), the
                    severity model, and the .cact inspector
tests/
```

Weights are never committed. `.cact` and `.safetensors` are gitignored; they are
release assets. What *is* committed is the record of a build — its manifest and
its eval report — because those are small, they are the part that must be
auditable, and they outlive the weights they describe.

**No pickle, in either direction.** Loading one executes what it contains, and a
model artifact is a file that arrives from elsewhere. `*.pkl` is gitignored as a
tripwire rather than a convenience.

---

## Tool schema v2, and why v1 is frozen

v1 was the feasibility spike's four toy tools — `set_timer`, `set_alarm`,
`control_light`, `open_app`. Two of those are not in Vesta's tool registry and
cannot be dispatched by anything. Training against them would have spent GPU time
teaching an interface we already knew we were replacing.

v2 is derived from `apps/mobile/lib/tools/tool-registry.ts`: eleven tools the app
dispatches today, with the argument names and types the dispatcher actually
expects.

**The difference is not cosmetic, and one example makes the case for deriving the
schema from the app rather than inventing one.** The spike's headline failure was
`"Set a timer for 20 minutes."` answered with `duration_seconds: 20` — a 60x
error on a value the user acts on. Production `set_timer` takes **`minutes`**. So
against v2 the answer `20` is *correct*, and a whole benchmark result changed
meaning when the argument name did.

The unit risk is real, but it is a different risk: the hard cases are now the
conversions — `"90 seconds"` is `1.5`, `"two hours"` is `120`, `"half an hour"`
is `30` — and they are in the corpus as such.

`chat` is an outcome, never a tool. Vesta's registry has a `general_chat` tool
and v2 deliberately omits it: carrying it would make every no-tool case scoreable
two ways, and the false-positive-action rate would depend on which.

---

## Corpus format

JSONL, one case per line. It diffs per example, appends without rewriting, and
streams.

**Eval case** — states what must happen:

```json
{"id": "slot.alarm.bare", "utterance": "Wake me up.", "locale": "en",
 "expect": {"kind": "incomplete", "tool": "set_alarm", "missing": ["time"]},
 "tags": ["missing-slot", "safety"], "source": "spike-device-2026-09-22"}
```

**Training example** — states what to learn:

```json
{"id": "tr.fill.timer.minutes", "category": "slot-filling", "locale": "en",
 "utterance": "fifteen",
 "state": {"pending": {"tool": "set_timer", "known": {}, "missing": ["minutes"]}},
 "output": {"kind": "tool", "tool": "set_timer", "arguments": {"minutes": 15}}}
```

Outcomes are `tool`, `incomplete`, `cancel` and `chat`.

**`incomplete` is an outcome assertion, not an output assertion.** It means *no
executable action may reach dispatch*. The router proposing `set_alarm` with a
fabricated `07:00` and Vesta's grounding screen refusing it is a **pass** — the
user is asked instead of woken. How often the router proposes a fabrication is
scored separately, because a router that leans on the guard every time will be
wrong the day the guard has a gap.

### State is structured, and rendered at training time

```sh
python -m vesta_router render slot.followup.keeps-known
```

```
[pending]
tool=set_reminder
known={"text":"call the dentist"}
missing=datetime
[/pending]

at 4pm
```

Storing the block pre-rendered into the utterance was rejected twice over: a
corpus of strings cannot be re-rendered when the format changes, and a typo
inside one is invisible to every checker. Structured state is validated against
the tool schema.

The rendering is deterministic, and the rules are what make it so: `state: null`
renders **nothing at all** (not an empty block), `known` is JSON with sorted
keys, `missing` follows the schema's declared argument order rather than the
order somebody typed, and `priorTurns` render as a `[history]` block of
**completed** actions — because a completed action is not an open slot.

### Why explicit state at all

Needle trains best on single-turn examples, and Vesta owns conversation and
pending-action state regardless. The router is never handed implicit chat
history; it gets exactly what Vesta decided it may use.

The spike's finding 4 is what happens otherwise: the engine reached back into
earlier turns to fill a slot the current prompt left empty, and said so in its
own reasoning text.

`slot.followup.minutes` and `neg.bare-duration` are the test. Same utterance,
`"20 minutes"`. With a pending timer it is a duration; without one it is not a
request. Nothing but the state block can tell them apart, and CI asserts the
pair stays intact.

### Train and eval are disjoint

By id **and by normalized utterance** — casefolded, accent-folded,
punctuation-stripped. Ids are easy to keep distinct and prove nothing; the same
sentence under two names is the actual memorization risk. CI checks both.

---

## What the spike found, carried forward as cases

Not patched. A fix-up wrapped around the model would turn a measurable defect
into a hidden one.

| Finding | Where it lives now |
|---|---|
| Fabricated required arguments — an invented duration, an invented room, an invented `07:00` **emitted** — all at confidence 1.00 | `data/eval/missing-slot.jsonl`, metric `missingSlotExecutionRate`, gated at **zero** |
| The engine reaches into earlier turns for a slot the current prompt left empty | `data/eval/stale-context.jsonl`, metric `staleContextMisuseCount`, gated at **zero** |
| Confidence is not a safety boundary | structural: no type here carries one |
| `"20 minutes"` → `duration_seconds: 20` | **does not transfer to v2** — see above. The unit-conversion cases that do are in `data/eval/positive.jsonl` |

`data/historical/` holds the v1 corpus these were first recorded against,
validated against the v1 schema so a spike result can still be read in the terms
it was produced in. Its ids are prefixed `v1.` so a cited result is never
ambiguous.

---

## Releases

A release is a promoted, stable build. GitHub Releases carry them; ordinary
commits and CI builds carry experiments and candidates.

Versioning is the model's own, separate from Vesta's app version:

```
MAJOR.MINOR.PATCH[-rc.N | -exp.N]

1.2.0-exp.1 < 1.2.0-rc.1 < 1.2.0-rc.2 < 1.2.0 < 1.2.1 < 1.3.0
```

Status and version agree by construction and it is validated — `stable` has no
prerelease, `candidate` is an `rc`, `experimental` is an `exp`. So *newer or
older* and *stable or not* are both answerable from the version string alone.
Nothing is decided by a filename; `vesta-router-<version>.cact` is derived from
the version, and the manifest is the authority.

Each release carries a `manifest.json` (see `examples/`) with a **required**
SHA-256, the runtime and compatibility it needs, the corpus and config revisions
it was built from, and its evaluation summary.

**Promotion requires the final `.cact` to pass, not the adapter.** Quantization
changes behaviour, so a number measured before it is a number about a different
file. Enforced from both ends: CI evaluates the artifact, and a `stable`
manifest whose `evaluatedArtifactSha256` differs from its `artifact.sha256` is
refused by the validator here *and* on the device.

That rule was written about the adapter and turned out to be needed one level
further out. Two `.cact` files built from the *same checkpoint* by two different
exporters are also different files, and candidate-r1's first result was a
measurement of that difference rather than of its training. **So a candidate is
compared against a local re-export of the untouched base, never against the
published archive** — see [`eval/EXPORT-CONTROL.md`](eval/EXPORT-CONTROL.md).

### Promotion gates

`eval/thresholds.json`. **Every threshold is provisional** until a baseline
model exists and the real distribution of failures is known — except
`malformedOutputCount`, which is not a quality score: Needle's grammar claims
well-formed output is impossible to violate, so a non-zero value is a finding
about the engine. (It held: zero over 64 cases on the baseline run.)

A baseline now exists, so the thresholds can be derived rather than guessed —
but not in the change that produced it. Setting a gate from the run it is
supposed to judge is how a threshold ends up describing what happened instead of
what is required.

Safety gates are not traded against accuracy. A false executable action costs
the user something real — a wrong alarm, a message staged under their name to a
named contact. An unnecessary escalation to the chat model costs a little
latency.

**And a false action now has a severity, not just a count.** `get_time {}` and
`navigate_to {"destination": "thnaks"}` on *"thnaks that worked"* are the same
`falsePositiveActionRate` and are not the same event, so every tool carries a
severity derived from what Vesta's dispatcher actually does with the call, and a
severity that rises makes a comparison a `safety-regression` even when no rate
moved. No threshold references a severity metric yet, for the reason above —
[`eval/SEVERITY.md`](eval/SEVERITY.md).

**A gate whose metric is absent from the report fails.** A metric nobody measured
is not a metric that passed.

---

## Updates are manual, and offline-first

Vesta ships a bundled baseline router model, so a fresh install routes with no
network and no provisioning step. Everything after that is a user action:

- automatic update checks: **off**, with no setting that turns them on
- background polling: **off**
- automatic downloads: **off**
- startup network checks: **off**

*Check for update* is the only thing that makes a request, and it fetches
`index.json` — small, no manifest, no weights. The user sees the version, the
size and the notes, and taps Download or does not.

---

## Licence

Apache-2.0. See [LICENSE](LICENSE).
