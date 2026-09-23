"""Fixed behaviour probes for the reasoning-alignment diagnostic.

DIAGNOSTIC ONLY. Not an evaluator, not a gate, and nothing it prints is a
metric of record. It exists to answer one question: does a fine-tune's
act / don't-act behaviour reach the normal Needle serving path?

TWO WAYS TO RUN THE SAME PROBE

``native``  the ``.cact`` through the runtime the evaluator uses
            (``vesta_router.engines.needle.NeedleEngine``: the cactus-needle
            3.0.1 wrapper and ``libneedle.so`` 3.0.1). This is the serving path.
``python``  the same weights BEFORE export: base checkpoint, optional LoRA
            merged, CQ W4 straight-through weights and A8 activations
            (``quant=True``) - the numerics training optimises. Greedy decode
            of an explicitly rendered prompt, so the assistant opener is a
            parameter instead of something the engine decides.

Both are scored by :func:`vesta_router.evaluate.score_case`, so "guarded
outcome" means exactly what it means in an eval report.

The python path has none of the engine's post-processing (grammar-constrained
decoding, the argument repair step, confidence gating). A B/C difference can
therefore come from export numerics OR from that machinery; the report keeps
both raw texts so the two can be told apart.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("NEEDLE_TELEMETRY", "0")
os.environ.setdefault("DO_NOT_TRACK", "1")

from vesta_router.corpus import Case, load_corpus  # noqa: E402
from vesta_router.engines import tool_definitions  # noqa: E402
from vesta_router.engines.needle import NEEDLE_SYSTEM_PROMPT  # noqa: E402
from vesta_router.evaluate import Call, EngineResponse, score_case  # noqa: E402
from vesta_router.render import render_example  # noqa: E402
from vesta_router.schema import load_tool_schema  # noqa: E402

SCHEMA_PATH = ROOT / "tools/tool-schema-v2.json"
PROBES_PATH = Path(__file__).with_name("probes.json")
NEEDLE_301 = Path.home() / "working/needle-f189b23"


def load_probes(path: Path = PROBES_PATH) -> list[tuple[dict, Case]]:
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    cases, problems = load_corpus(ROOT / "data/eval")
    if problems:
        raise SystemExit(f"eval corpus does not load: {problems[0]}")
    by_id = {case.id: case for case in cases}
    return [(probe, by_id[probe["evalId"]]) for probe in spec["probes"]]


def verdict(case: Case, response: EngineResponse, schema) -> dict:
    result = score_case(case, response, schema)
    expected = case.outcome
    if expected["kind"] == "tool":
        passed = result.full_call_exact and result.executable_action
        raw_passed = result.full_call_exact
    else:
        passed = result.guarded_outcome == "no_executable_action"
        raw_passed = result.raw_outcome == "no_call"
    return {
        "expected": {k: expected[k] for k in ("kind", "tool", "arguments") if k in expected},
        "reasoning": response.reasoning,
        "functionCalls": [c.as_json() for c in response.function_calls],
        "suppressedCalls": [c.as_json() for c in response.suppressed_calls],
        "rawOutcome": result.raw_outcome,
        "guardedOutcome": result.guarded_outcome,
        "rawPass": raw_passed,
        "pass": passed,
        "error": response.error,
    }


# ── native (.cact, serving path) ──────────────────────────────────────────


def run_native(cact: str, probes: Path = PROBES_PATH) -> dict:
    from vesta_router.engines.needle import NeedleEngine

    schema = load_tool_schema(SCHEMA_PATH)
    engine = NeedleEngine(cact, tool_definitions(SCHEMA_PATH), needle_path=str(NEEDLE_301))
    rows = []
    for probe, case in load_probes(probes):
        engine.reset()
        response = engine.complete(render_example(case.utterance, case.state, schema))
        row = {"probe": probe["key"], "evalId": case.id, "utterance": case.utterance}
        row.update(verdict(case, response, schema))
        row["confidence"] = (response.raw or {}).get("confidence")
        rows.append(row)
    engine.close()
    return {"stage": "native", "engine": engine.describe(), "rows": rows}


# ── python (pre-export weights, explicit prompt) ──────────────────────────

OPENERS = {"none": "", "think": "<think>\n"}


def _parse(text: str) -> tuple[str | None, tuple[Call, ...], str | None]:
    # Greedy decoding without the engine's stop logic runs on past the turn
    # into invented follow-up turns; only the first assistant turn is the answer.
    text = text.split("<|im_end|>", 1)[0]
    reasoning = None
    if "<think>" in text:
        reasoning = text.split("<think>", 1)[1].split("</think>", 1)[0].strip()
    if "<tool_call>" not in text:
        return reasoning, (), "no <tool_call> block"
    body = text.split("<tool_call>", 1)[1].split("</tool_call>", 1)[0]
    try:
        calls = json.loads(body)
    except json.JSONDecodeError as exc:
        return reasoning, (), f"unparseable tool_call JSON: {exc}"
    if not isinstance(calls, list):
        return reasoning, (), "tool_call is not a list"
    parsed = tuple(
        Call(
            name=str(c.get("name", "")),
            arguments=c.get("arguments") if isinstance(c.get("arguments"), dict) else {},
        )
        for c in calls
        if isinstance(c, dict)
    )
    return reasoning, parsed, None


def run_python(
    adapter: str | None, opener: str, tools_format: str, probes: Path = PROBES_PATH, max_new: int = 160
) -> dict:
    import jax
    import jax.numpy as jnp
    import numpy as np
    from needle.model.architecture import SimpleAttentionNetwork
    from needle.model.checkpoints import read_adapter
    from needle.model.finetune import merge_lora
    from needle.model.finetune import render_example as needle_render
    from needle.model.quantize import WEIGHT_BITS, configure_deploy, cq_ste_params
    from needle.model.run import load_checkpoint
    from needle.model.tokenizer import BOS_ID, EOS_ID, IM_END_ID, PAD_ID, get_tokenizer

    schema = load_tool_schema(SCHEMA_PATH)
    tools = tool_definitions(SCHEMA_PATH)
    if tools_format == "compact":
        tools_field = tools  # the trainer serialises a list with separators=(",", ":")
    else:
        tools_field = json.dumps(tools, ensure_ascii=False)  # what r1/r2 training embedded

    params, config = load_checkpoint(str(ROOT / "checkpoints/needle3.safetensors"))
    config.dtype = "float32"
    params = jax.tree.map(lambda a: np.asarray(a).astype(np.float32), params)
    if adapter:
        blob = read_adapter(adapter)
        lora = {
            tuple(k.split("/")): {"A": jnp.asarray(v["A"]), "B": jnp.asarray(v["B"])}
            for k, v in blob["lora"].items()
        }
        params = merge_lora(params, lora, blob["scale"])
    configure_deploy(act_bits=getattr(config, "act_bits", 8), kv_bits=getattr(config, "kv_bits", 8))
    merged = jax.device_put(cq_ste_params(params, WEIGHT_BITS))
    model = SimpleAttentionNetwork(config)
    tokenizer = get_tokenizer(config.vocab_size)

    @jax.jit
    def logits_at(p, buf, pos):
        return model.apply({"params": p}, buf, quant=True)[0, pos]

    rows = []
    started = time.perf_counter()
    for probe, case in load_probes(probes):
        query = render_example(case.utterance, case.state, schema)
        prompt, _ = needle_render(
            {"system": NEEDLE_SYSTEM_PROMPT, "tools": tools_field, "query": query, "answers": []}
        )
        prompt += OPENERS[opener]
        ids = [BOS_ID] + tokenizer.encode(prompt)
        buf_len = -(-(len(ids) + max_new) // 128) * 128
        buf = (
            jnp.full((1, buf_len), PAD_ID, dtype=jnp.int32)
            .at[0, : len(ids)]
            .set(jnp.asarray(ids, dtype=jnp.int32))
        )
        generated = []
        for pos in range(len(ids) - 1, len(ids) - 1 + max_new):
            nxt = int(jnp.argmax(logits_at(merged, buf, pos)))
            if nxt == EOS_ID:
                break
            generated.append(nxt)
            if nxt == IM_END_ID:
                break
            buf = buf.at[0, pos + 1].set(nxt)
        text = OPENERS[opener] + tokenizer.decode(generated)
        text = text.split("<|im_end|>", 1)[0] + ("<|im_end|>" if "<|im_end|>" in text else "")
        reasoning, calls, problem = _parse(text)
        response = EngineResponse(raw={"text": text}, function_calls=calls, reasoning=reasoning)
        row = {"probe": probe["key"], "evalId": case.id, "utterance": case.utterance}
        row.update(verdict(case, response, schema))
        row["rawText"] = text
        row["parseProblem"] = problem
        rows.append(row)
    return {
        "stage": "python",
        "adapter": adapter,
        "adapterSha256": hashlib.sha256(Path(adapter).read_bytes()).hexdigest() if adapter else None,
        "opener": OPENERS[opener],
        "toolsFormat": tools_format,
        "numerics": f"CQ W{WEIGHT_BITS} STE weights + A8 activations (quant=True)",
        "backend": f"{jax.default_backend()} {jax.devices()[0]}",
        "wallS": round(time.perf_counter() - started, 1),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    native = sub.add_parser("native")
    native.add_argument("--cact", required=True)
    python = sub.add_parser("python")
    python.add_argument("--adapter")
    python.add_argument("--opener", choices=sorted(OPENERS), required=True)
    python.add_argument("--tools-format", choices=["compact", "spaced"], default="compact")
    for p in (native, python):
        p.add_argument("--probes", default=str(PROBES_PATH))
        p.add_argument("--label", required=True)
        p.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.mode == "native":
        report = run_native(args.cact, Path(args.probes))
    else:
        report = run_python(args.adapter, args.opener, args.tools_format, Path(args.probes))
    report["label"] = args.label
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    passed = sum(r["pass"] for r in report["rows"])
    print(f"{args.label}: {passed}/{len(report['rows'])} pass")
    for r in report["rows"]:
        calls = json.dumps(r["functionCalls"], ensure_ascii=False)
        print(f"  {'PASS' if r['pass'] else 'FAIL'} {r['probe']:<22} {r['guardedOutcome']:<22} {calls}")
        print(f"       reasoning: {r['reasoning']!r}")


if __name__ == "__main__":
    main()
