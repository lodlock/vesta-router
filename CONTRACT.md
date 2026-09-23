# Which side owns what

The authoritative contract is in the Vesta repository at
[`docs/router-model-contract.md`](https://github.com/lodlock/vesta/blob/main/docs/router-model-contract.md).
This file is the boundary, stated from this side, so that a change made here has
an obvious answer to "does the app need to change too?".

The seam between the two repositories is exactly **two documents** — a
**manifest** and an **update index** — and Vesta validates both rather than
trusting either.

---

## This repository owns

| | |
|---|---|
| Training and evaluation corpora | `data/` |
| Tool schemas | `tools/` — v1 frozen, v2 active |
| The state-block rendering rules | `src/vesta_router/render.py` |
| Training configuration and scripts | `train/` |
| The eval harness and its report format | `src/vesta_router/{evaluate,grounding,runner,engines}`, `eval/`, `reports/`, `examples/eval-report-*.json` |
| The evaluator-side grounding screen | `src/vesta_router/grounding.py` — an approximation of Vesta's, labelled as one in every report; Vesta's own guard stays Vesta's |
| Promotion gates | `eval/thresholds.json` |
| Building the `.cact` and writing its manifest | `train/`, `releases/` |
| Publishing releases and `index.json` | GitHub Releases |

## Vesta owns

| | |
|---|---|
| Deterministic grounding | `lib/scheduling/grounding.ts` and the general router screen |
| Missing-slot enforcement | refuses an action whose required argument has no evidence |
| Unit and format normalization | at dispatch, from the user's own tokens |
| Conversation and pending-action state | and decides what of it the router may see |
| Execution authorization | the confirm gate, the tool dispatcher |
| Manifest validation on device | `lib/router-model/manifest.ts` |
| Compatibility screening | `lib/router-model/compat.ts` |
| The update policy and its network triggers | `lib/router-model/update-policy.ts` |
| Activation and rollback | `lib/router-model/slots.ts` |
| Hosting `:router` and killing it | the AIDL service, the JNI bridge |

**Model confidence belongs to neither side.** It is produced by the engine and
consumed by nothing: there is no parameter for it in either repository, and a
fabricated `07:00` arriving at 1.00 is why.

---

## Deliberate duplication, and how it is kept honest

Three rules exist in both repositories, in two languages. That is a cost, paid on
purpose: the app must decide "is this newer, and is it stable" with no Python
available, and the release pipeline must decide the same thing with no
TypeScript.

| Rule | Here | In Vesta |
|---|---|---|
| Version scheme and ordering | `src/vesta_router/version.py` | `lib/router-model/version.ts` |
| Manifest validation | `src/vesta_router/manifest.py` | `lib/router-model/manifest.ts` |
| Promotion invariant (`evaluatedArtifactSha256 == artifact.sha256`) | both | both |

They are kept in step by asserting the **same cases** on both sides —
`tests/test_version.py` and `version.test.ts` check the same ordering chain,
`tests/test_manifest.py` and `manifest.test.ts` the same refusals.

**CI's validator is deliberately stricter than the device's**, in one direction
only: it additionally requires the artifact filename to be the derived
`vesta-router-<version>.cact`, and refuses an unrecognised metric name. A rule
that fails here and passes there costs a release author a fix; the reverse would
be a release that shipped and cannot be installed.

---

## What has to change together

| A change to | Requires |
|---|---|
| The tool schema (a new tool, a renamed argument) | a new `toolSchemaVersion`, a MAJOR model version, and a Vesta release that dispatches it |
| The manifest shape | a new `manifestVersion`, and a Vesta build that reads it |
| The `:router` IPC envelope | a new `requiredRouterProtocolVersion` on both sides |
| The state-block rendering | retraining, and Vesta's future router caller must render identically |
| A threshold in `eval/thresholds.json` | nothing in Vesta — gates are a release decision |
| A corpus case | nothing in Vesta |

The last two rows are the point of the split: most of the work here ships
without touching the app at all.
