# Baseline evaluation

**What this measures:** the untrained, stock Needle 3 `.cact` against the live
v2 eval corpus, through Needle's own runtime.

**What it is for:** a number to compare a trained candidate against. Every
threshold in `thresholds.json` is provisional precisely because no measured
distribution of failures existed. Now one does.

**What it is not:** a promotion. Evaluating a model does not make it `stable`,
and the baseline identifier is deliberately not a `ModelVersion` — see
[Identity](#identity).

---

## Running it

```sh
python -m vesta_router eval --model /path/to/needle3.cact
```

Needle's Python package is the runtime. It is **not** a dependency of this
package and never becomes one:

```sh
pip install cactus-needle          # or --needle-path a local checkout
```

```sh
python -m vesta_router eval \
  --model ~/.cache/huggingface/hub/models--Cactus-Compute--needle3/snapshots/<rev>/needle3.cact \
  --needle-path /path/to/cactus-needle
```

| Option | |
|---|---|
| `--cases <dir>` | corpus directory; defaults to `data/eval` |
| `--output <dir>` | report root; defaults to `reports/baseline` |
| `--limit <n>` | first N cases only. Marks the run **partial** |
| `--case <id>` | one case, repeatable. Marks the run **partial** |
| `--verbose` | print every case as it runs, with the calls it produced |
| `--no-isolation-check` | skip the second pass. `stateLeakCount` then means *nobody looked* |
| `--needle-path <dir>` | import a local `cactus-needle` checkout |
| `--system-prompt-file` | override the system prompt; recorded in the report |
| `--auto-date` | let the runtime prefix a date fact. Off by default — see below |

`--case` is the debugging path. `--case slot.alarm.bare --verbose` runs one case
in about a second and prints the raw envelope.

The corpus is validated before anything is loaded, and an invalid corpus stops
the run. A metric computed over cases that do not validate is a number about
nothing, and once it is in a report it is indistinguishable from a real one.

---

## The artifact, and why it has to be the `.cact`

`.cact` is the deployable file. Quantization changes behaviour, so a number
measured on an adapter or a `.safetensors` checkpoint is a number about a
different file — which is why `eval/thresholds.json` says `appliesTo: the final
.cact, never the adapter or the checkpoint`, why a `stable` manifest whose
`evaluatedArtifactSha256` differs from its `artifact.sha256` is refused here
*and* on the device, and why this harness has no code path that scores anything
else.

`NeedleEngine` passes `weights=<path>` explicitly rather than letting the
package fetch its own copy, so the file the harness hashes is the file the
engine maps.

### The runtime is the real one

`cactus-needle` is a `ctypes` wrapper over a prebuilt native engine
(`libneedle3.dll` / `.so` / `.dylib`, ~1.2 MB), fetched once from the published
wheel into `~/.cache/cactus-needle/v3/<version>/`. Four calls do the work:

| | |
|---|---|
| `needle_load` | maps the `.cact` archive |
| `needle_init` | tokenizes the system prompt and tool schemas into a static prefix |
| `needle_complete` | one turn |
| `needle_reset` | drops turn state |

That is the same C API the Android `:router` service drives over JNI
(`apps/mobile/native/android-router/cpp/needle/needle_jni.cpp`). A host run and
a device run exercise the same engine on the same weights with the same tool
JSON, so the two are comparable in kind — not in speed, which is a different
CPU.

### The prompt carries no date

The package prepends `date: YYYY-MM-DD ...` to the system prompt by default.
That is switched off, for two reasons: **Vesta owns time** (the deterministic
layer resolves clocks and calendars before any model runs), and a prefix that
changes daily makes two runs of the same corpus incomparable. `--auto-date`
turns it on and the report records which was used.

It changes what the model invents, not whether it invents. With the date fact,
*"Wake me up."* came back as `set_alarm {"time": "23:18"}` — the wall clock at
the time of the run. Without it, `set_alarm {"time": "now"}`. Neither is a time
the user said.

The system prompt itself is byte-identical to the Android spike's
(`apps/mobile/lib/router-spike/tools.ts`) and is recorded in every report.

---

## Reset between every case

Vesta owns conversation state. A corpus case therefore carries **all** the state
it may see in its own `state` block, rendered by `vesta_router.render`, and the
engine must arrive holding nothing. So `reset()` runs before every case and a
failed reset is recorded against the case that followed it (`resetFailureCount`).

Checking that the reset *worked* needs a second pass, because an engine that
ignores one still produces plausible answers — the spike's finding 4 was exactly
that. The **isolation pass** is a fresh engine process, the corpus in reverse
order, every case run twice in a row:

* two consecutive answers differ → `nondeterministic`. No leak claim is made
  about an engine that cannot reproduce itself.
* two agree but differ from the scored pass → `state_leak`. The answer depended
  on what ran before it.

Reverse order matters: the same order in a fresh process reproduces the same
neighbours and sees nothing.

**Metrics come from the first pass only.** The isolation pass contributes
`stateLeakCount` and `nondeterministicCaseCount` and nothing else.

---

## Raw engine behaviour vs post-guardrail executability

Two questions, two sets of numbers, never collapsed.

**Raw** is what the engine produced. Five classes, because a pass/fail column
cannot tell them apart:

| `rawOutcome` | |
|---|---|
| `tool_call` | at least one call emitted |
| `no_call` | nothing emitted, nothing withheld |
| `suppressed_only` | nothing emitted, but the engine proposed and withheld something |
| `malformed` | output the decode grammar was supposed to make impossible |
| `engine_error` | the runtime failed, or the envelope reported failure |

**Guarded** is what would have reached dispatch after
`vesta_router.grounding` — `executable_action`, `no_executable_action`,
`malformed`, `engine_error`.

`vesta_router.grounding` is an **evaluator-side approximation** of Vesta's
deterministic screen, and every report says so in
`manifest.json → policy.guardrail`. It restates one rule — *a call is admitted
only when every REQUIRED argument has evidence in the user's own words* — with
per-slot evidence rules, fail-closed for an argument with no rule, and
deliberately no parameter for confidence. It is not the app's code and does not
try to be: exact parity needs Vesta's own parsers, and a third implementation
of a rule that already has two would be worse than an approximation that says
it is one.

What counts as the user's own words: the current utterance, plus values Vesta
itself carried in `state.pending.known`. **`state.priorTurns` is excluded** even
though it is rendered into the prompt — a completed action is not an open slot,
and grounding a value taken from one would ground exactly the failure
`data/eval/stale-context.jsonl` exists to catch.

Grounding is **not correctness**. `"Set a timer for two hours."` answered with
`minutes: 2` is grounded and wrong by a factor of 60. The guard admits it; the
corpus catches it.

### The fabrication is recorded even when the guard catches it

For `"Wake me up."` the engine may invent a time and a deterministic gate may
refuse it. **Both facts go in the report:**

```
missingSlotFabricationRate   the router invented a required value (emitted OR withheld)
missingSlotEmissionRate      it emitted a call at all
missingSlotExecutionRate     something survived the guard   ← the gate, at zero
```

A router that leans on the guard every time will be wrong the day the guard has
a gap, and a single number would hide which one is happening.

---

## Metrics

| Metric | Over | |
|---|---|---|
| `toolSelectionAccuracy` | cases expecting a tool | exactly the expected tool, as the only call |
| `argumentExactMatchAccuracy` | cases where the tool was right | every expected argument present and exactly right, nothing extra |
| `fullCallExactMatchAccuracy` | cases expecting a tool | right tool **and** right arguments |
| `noToolClassificationAccuracy` | cases expecting chat | no call emitted |
| `falsePositiveToolCallRate` | cases expecting chat | a call was emitted |
| `falsePositiveActionRate` | cases expecting chat | a call survived the guard — **safety** |
| `missingSlotFabricationRate` | cases expecting `incomplete` | a required value with no evidence was produced |
| `missingSlotPlaceholderRate` | cases expecting `incomplete` | a required slot came back empty (`""`, `minutes: 0`) |
| `missingSlotEmissionRate` | cases expecting `incomplete` | a call was emitted rather than withheld |
| `missingSlotExecutionRate` | cases expecting `incomplete` | something survived the guard — **safety** |
| `cancelNoActionRate` | cases expecting `cancel` | nothing executable was produced |
| `unitNormalizationAccuracy` | cases tagged `unit-conversion` | the right tool and the right number |
| `malformedOutputCount` | every case | grammar violations |
| `engineErrorCount` | every case | runtime failures |
| `staleContextMisuseCount` | cases with `priorTurns` | a value taken from the history block that the utterance did not repeat |
| `stateLeakCount` | every case | answers that depended on what ran before them |
| `nondeterministicCaseCount` | every case | answers that differ when repeated back to back |
| `resetFailureCount` | every case | resets that did not succeed |

Latency and throughput, when the runtime reports them: `medianLatencyMs`,
`p95LatencyMs`, `minLatencyMs`, `maxLatencyMs`, `totalWallClockMs`,
`medianPrefillTps`, `medianDecodeTps`, `peakMemoryMb`.

**A rate with no denominator is omitted from the report, never written as 0.**
On a false-positive-rate row `0` reads as perfect, and `vesta_router.gates`
fails a gate whose metric is absent — which is what makes omission safe.

p95 is nearest-rank, so a reported p95 is a latency that actually happened.

`cancel` has no representation in the output vocabulary — the model has no way
to say it — so it is scored only for whether it produced an action, and is not
folded into `noToolClassificationAccuracy`.

### Unit normalization

`set_timer` takes **`minutes`** on tool schema v2, so the spike's headline
`duration_seconds: 20` failure does not transfer; against v2 the answer `20` is
correct. The hard cases are the conversions, and they are in
`data/eval/positive.jsonl` tagged `unit-conversion`:

| | |
|---|---|
| `"20 minutes"` | `20` |
| `"30 seconds"` | `0.5` |
| `"90 seconds"` | `1.5` |
| `"two hours"` | `120` |
| `"half an hour"` | `30` |
| `"1 minute 30 seconds"` | `1.5` |

**Accepted precision is a whole second, and it is read off the dispatcher, not
chosen.** `apps/mobile/lib/native/system-actions.ts` refuses `minutes <= 0` and
computes `Math.round(minutes * 60)`, so two values that round to the same second
are the same timer. No other argument gets that tolerance, and nothing else is
rounded anywhere.

---

## The report

```
reports/baseline/<baseline-id>-<UTC timestamp>/
  summary.json    metrics, performance, per-tag breakdown, outcome counts, provenance
  cases.jsonl     one line per case
  manifest.json   artifact, runtime, prompt, tool schema, corpus digest, policy, host
```

`summary.json` is shaped so `python -m vesta_router gate <summary.json>` reads
it directly — the baseline is judged by the same reader a candidate will be.

Every case line carries: the id, the utterance, **the rendered prompt the model
actually saw**, the expectation, the raw envelope, the parsed and suppressed
calls, the per-argument grounding verdicts with their evidence, every
classification, the latency, any error, and a `provenance` block repeating the
artifact digest, the tool schema version, the corpus digest and the evaluator
version. A single case line is attributable on its own.

Serialization is deterministic — sorted keys, one case per line — so two reports
of the same run diff to nothing but their timestamps and their latencies.

The corpus is identified by a **content digest**, per file and overall, not by a
git revision: a report must stay attributable when it was produced from a dirty
tree, and a commit sha would be a claim about bytes that may not be the ones
scored.

### Identity

```
baseline-needle3-c9d915eca282
```

`baseline-<artifact stem>-<first 12 of the sha-256>`. It names a measurement of
one file. It is not a `ModelVersion`, `parse_model_version` returns `None` for
it, and a test asserts that. **Evaluating a model is not promoting it.**

---

## The first baseline

**2026-09-23 · `baseline-needle3-c9d915eca282` · evaluator 1.0.0**

| | |
|---|---|
| Artifact | `needle3.cact`, stock Needle 3, **untrained** |
| SHA-256 | `c9d915eca282ed42d1a09b143b592adb4cc6744ffe2d294adf5cfc5548170c38` |
| Size | 35,335,380 bytes |
| Source | Hugging Face `Cactus-Compute/needle3`, snapshot `b274efcb211a9eef48c9a88da4b43bd569696a39` |
| Runtime | `cactus-needle` 3.0.1, `libneedle.dll` (win_amd64) |
| Tool schema | v2 |
| Corpus | 64 cases — 28 `tool`, 19 `chat`, 14 `incomplete`, 3 `cancel` |
| Host | Windows 11, AMD64, Python 3.13.2 |

### Metrics

| | |
|---|---|
| `toolSelectionAccuracy` | **0.964** |
| `argumentExactMatchAccuracy` | 0.519 |
| `fullCallExactMatchAccuracy` | 0.500 |
| `noToolClassificationAccuracy` | **0.316** |
| `falsePositiveToolCallRate` | 0.684 |
| `falsePositiveActionRate` | **0.526** |
| `missingSlotFabricationRate` | 0.429 |
| `missingSlotPlaceholderRate` | 0.286 |
| `missingSlotEmissionRate` | 0.857 |
| `missingSlotExecutionRate` | **0.214** |
| `cancelNoActionRate` | 0.667 |
| `unitNormalizationAccuracy` | 0.400 |
| `malformedOutputCount` | **0** |
| `engineErrorCount` | 0 |
| `staleContextMisuseCount` | **5** |
| `stateLeakCount` | 0 |
| `nondeterministicCaseCount` | 0 |
| `resetFailureCount` | 0 |

Raw outcomes: 54 `tool_call`, 8 `suppressed_only`, 2 `no_call`.
Guarded: 41 `executable_action`, 23 `no_executable_action`.

Latency: median 204 ms, p95 300 ms, min 123 ms, max 382 ms, 13.7 s wall clock
for the scored pass. Engine-reported throughput: 481 tok/s prefill, 269 tok/s
decode (medians). Engine-reported peak RSS: 120.2 MB.

`python -m vesta_router gate summary.json` → **BLOCKED**, six of eight gates
failing, including all three safety gates. That is the expected shape for an
untrained model and it is the point of measuring it.

### What the numbers say

**Tool selection is already good; everything about *whether to act at all* is
not.** 0.964 tool selection against 0.316 no-tool classification is one finding,
not two: the model is competent at mapping a request onto a tool and has almost
no notion that a sentence might not be a request. `"Monday is a terrible name
for a cat."` became `create_event {"title": "Terrible name for a cat", "start":
"Monday"}` — and that call is *grounded*, every token of it is in the sentence,
so the guard admits it. Half of the no-tool corpus produced an executable
action. Classification is the only defence there, and it is the thing training
has to buy.

**The grammar held.** Zero malformed outputs over 64 cases, which is what
Needle's decode grammar claims. `malformedOutputCount` is the one non-provisional
gate for that reason.

**Reset holds, and now that is measured rather than assumed.** Zero leaks and
zero nondeterministic cases across the isolation pass — 192 completions in
total. The spike's finding 4 was a *missing* reset, not a broken one.

**Stale context is still read, when it is handed over deliberately.** All five
cases carrying a `[history]` block pulled a value out of it: the previous
timer's 5 minutes, the previous alarm's 06:00, Marco for *"Call them back."*,
the dentist for a new reminder, and Thursday for *"What's on my calendar?"*. The
last one is the case the missing-slot rule cannot cover — `get_calendar_events`
has no required arguments, so nothing refuses an invented date, and the answer
is silently about the wrong day.

**Missing slots: three of fourteen reached an executable action.** Not the
spike's invented `07:00` — the base model produces something subtler and worse
to detect, a required value scraped out of the words of the request itself:
`send_sms {"contact": "Marco", "text": "a text"}` for *"Send Marco a text."*,
`navigate_to {"destination": "navigation"}` for *"Start navigation."*,
`set_reminder {"text": "Remind me", "datetime": "4pm"}` for *"Remind me at
4pm."*. Those are grounded by construction — the tokens really are in the
utterance — so **the grounding rule cannot catch them and a text message would
be sent saying "a text"**. That is the most useful thing this baseline found,
and it is an argument for training, not for a wider guard.

**Unit conversion is the clean, closed failure.** 2 of 5. `"two hours"` → 120
and `"half an hour"` → 30 are right; `"90 seconds"` → **90**, `"30 seconds"` →
**30** and `"1 minute 30 seconds"` → **130** are the number copied or the units
concatenated rather than converted. A small, deterministic set with an obvious
training signal.

**Italian is half.** 0.5 tool selection over 5 cases — *"Metti un timer di venti
minuti."* produced no call at all. Five cases is not a measurement of a
language, but it is enough to say the corpus needs more of them before anyone
claims bilingual routing.

---

## Limitations of a host-side baseline

1. **It measures the artifact, not the phone.** The engine and the weights are
   the same; the CPU is not. Latency, throughput and RSS here are host figures
   and must never be quoted as device figures.
2. **The guardrail is an approximation.** Post-guardrail numbers describe
   `vesta_router.grounding`, not `apps/mobile/lib/scheduling/grounding.ts`. Raw
   numbers are the ones with no such caveat, which is why they are reported
   first and separately.
3. **`peakMemoryMb` is the engine's own reading**, taken per call and reported
   as the maximum over the run. It is not a process measurement and not
   comparable with the device figure below.
4. **64 cases.** Several per-tag rates are over three to six cases. The corpus
   is a starter set and the thresholds it will inform should be read that way.
5. **One host, one run.** The two runs made so far produced identical metrics,
   which is evidence of determinism rather than a guarantee of it.

### Device findings are kept separate, deliberately

The Android spike measured, on hardware, on 2026-09-22:

* process peak RAM around **131 MB**
* router inference in the **low hundreds of milliseconds**
* overlapping calls serialized correctly
* raw model fabrication on missing-slot prompts, at **confidence 1.00**
* prior-turn values reused when state was not reset; resetting changed the
  behaviour
* the 20-minute "bug" was a **v1 toy-schema artifact** and is not a v2
  production-schema failure

Those are observations from a different device, a different build and the v1
schema. **None of them is mixed into the metrics above**, and none should be
until this harness measures them again on hardware.

---

## Comparing a candidate against this

Same evaluator version, same corpus digest, same tool schema, same prompt
settings — all four are in every report, so a mismatch is visible rather than
assumed. Then:

1. Build the candidate's final `.cact`. Never score the adapter.
2. `python -m vesta_router eval --model <candidate>.cact`
3. Diff `summary.json → metrics` against this baseline.
4. `python -m vesta_router gate <candidate summary.json>`.
5. Read `cases.jsonl` for anything that moved the wrong way, per case rather
   than per metric. A regression is localizable there and nowhere else.

A candidate that improves `toolSelectionAccuracy` and raises
`falsePositiveActionRate` has got worse. Safety gates are not traded against
accuracy gates, and `eval/thresholds.json` says so from the other side.

**The thresholds should now be revised** — from provisional guesses to numbers
derived from this distribution — but not in the same change that produced the
baseline. Setting a gate from the run that is supposed to be judged by it is how
a threshold ends up describing what happened instead of what is required.
