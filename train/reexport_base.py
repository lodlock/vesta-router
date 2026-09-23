"""Re-export the UNMODIFIED base checkpoint through candidate-r1's own exporter.

WHY THIS EXISTS

candidate-r1 was measured against the PUBLISHED base `.cact` - a file produced
by Cactus's own pipeline, not by the exporter that wrote the candidate. So
every delta between the two reports carried two causes at once: the training,
and whatever the two build paths do differently. This script removes the second
one by exporting the untouched checkpoint down candidate-r1's path, so a
three-way comparison can attribute a delta to one or the other.

WHY IT IS NOT JUST `needle build checkpoints/needle3.safetensors --out x.cact`

Because that command does not export anything. `needle.model.finetune.build_main`
short-circuits when there is no `--lora` and no depth change::

    if not args.lora and (not layers or layers == read_layers(base_archive)):
        shutil.copyfile(base_archive, out)
        print("... the published base archive")

It copies the published archive and says so. Verified: the output is
byte-identical to the published `.cact` (sha256 c9d915eca282...). The checkpoint
is never read. So the obvious "rebuild the base with the same command" is not a
control at all - it reproduces the very file it was supposed to control for.

WHAT THIS DOES INSTEAD

Exactly what `build_main` does on the branch candidate-r1 took, minus the LoRA
merge::

    load_checkpoint -> drop the confidence head -> rung(..., None)
                    -> write_export(bits=WEIGHT_BITS,
                                    tokenizer=<published archive's blob>,
                                    kv_window=effective_kv_window(config))

No adapter is read and none is applied. The confidence head is dropped because
candidate-r1's export dropped it, and a control that differs from the candidate
in its tensor set would not be a control.

AND IT PROVES THAT CLAIM RATHER THAN ASSERTING IT

With `--cross-check`, the same artifact is built a second time through the real
`needle build` CLI with a ZERO-VALUED adapter: candidate-r1's exact command, its
exact LoRA key set, rank and scale, with every B set to zero so the merged delta
is exactly `scale * (A @ 0) == 0`. If the two files are byte-identical then the
no-adapter export IS what candidate-r1's command produces on a null training
delta, and the control is demonstrated rather than argued.

NO PICKLE, in either direction. The checkpoint and the zero adapter are
safetensors, which is what `needle.model.checkpoints` writes for that suffix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = REPO / "checkpoints" / "needle3.safetensors"
ARTIFACTS = REPO / "artifacts"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def export_unmodified_base(checkpoint: Path, out: Path) -> dict:
    """`build_main`'s export branch with no adapter. Returns what it wrote."""
    os.environ.setdefault("NEEDLE_TELEMETRY", "0")
    os.environ.setdefault("DO_NOT_TRACK", "1")

    from needle.agent import fetch
    from needle.model.architecture import ConfidenceHead, effective_kv_window
    from needle.model.export import read_tokenizer_blob, write_export
    from needle.model.finetune import rung
    from needle.model.quantize import WEIGHT_BITS
    from needle.model.run import load_checkpoint

    # force=True is what build_main passes, so the tokenizer blob packed here is
    # the published one, exactly as candidate-r1's was.
    base_archive = fetch.fetch_weights(3, force=True)
    params, config, _ = load_checkpoint(str(checkpoint), return_run=True)

    dropped = ConfidenceHead.key in params
    if dropped:
        params = {k: v for k, v in params.items() if k != ConfidenceHead.key}

    params, config = rung(params, config, None)
    info = write_export(
        params,
        config,
        str(out),
        bits=WEIGHT_BITS,
        tokenizer=read_tokenizer_blob(base_archive),
        kv_window=effective_kv_window(config),
    )
    return {
        "baseArchive": str(base_archive),
        "baseArchiveSha256": sha256_file(Path(base_archive)),
        "confidenceHeadDropped": dropped,
        "layers": int(config.num_layers),
        "weightBits": int(WEIGHT_BITS),
        "tensors": int(info["tensors"]),
        "sizeBytes": int(info["bytes"]),
    }


def write_zero_adapter(reference: Path, out: Path) -> dict:
    """candidate-r1's adapter with every B zeroed: same keys, zero delta."""
    import numpy as np
    from needle.model.checkpoints import read_adapter, write_adapter

    adapter = read_adapter(str(reference))
    zeroed = {
        key: {"A": np.asarray(value["A"]), "B": np.zeros_like(np.asarray(value["B"]))}
        for key, value in adapter["lora"].items()
    }
    payload = dict(adapter)
    payload["lora"] = zeroed
    write_adapter(str(out), payload)
    return {
        "weightGroups": len(zeroed),
        "scale": adapter.get("scale"),
        "rank": adapter.get("rank"),
        "base": adapter.get("base"),
    }


def _needle_executable() -> str | None:
    """The `needle` console script, preferring the one beside this interpreter."""
    beside = Path(sys.executable).parent / ("needle.exe" if os.name == "nt" else "needle")
    if beside.is_file():
        return str(beside)
    return shutil.which("needle")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="re-export the unmodified base checkpoint")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--out-dir", default=str(ARTIFACTS))
    parser.add_argument(
        "--cross-check",
        metavar="ADAPTER",
        help="a reference adapter (candidate-r1's) to zero and rebuild through the real "
        "`needle build` CLI, proving the no-adapter export is what that command "
        "produces on a null training delta",
    )
    parser.add_argument("--record", help="write the build record JSON here")
    args = parser.parse_args(argv)

    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        print(f"{checkpoint}: no such checkpoint", file=sys.stderr)
        return 2
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    staging = out_dir / "local-base-reexport.staging.cact"
    print(f"  checkpoint {checkpoint}")
    print(f"             sha256 {sha256_file(checkpoint)}")
    info = export_unmodified_base(checkpoint, staging)
    digest = sha256_file(staging)
    final = out_dir / f"local-base-reexport-{digest[:12]}.cact"
    staging.replace(final)
    print(
        f"  wrote      {final}  {info['sizeBytes']} bytes  "
        f"{info['tensors']} tensors  W{info['weightBits']}A8"
    )
    print(f"             sha256 {digest}")

    record = {
        "artifact": {"path": str(final.relative_to(REPO)).replace("\\", "/"), "sha256": digest, **info},
        "checkpoint": {
            "path": str(checkpoint.relative_to(REPO)).replace("\\", "/"),
            "sha256": sha256_file(checkpoint),
        },
        "host": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
    }

    if args.cross_check:
        reference = Path(args.cross_check).resolve()
        zero = out_dir / "local-base-zero-adapter.safetensors"
        meta = write_zero_adapter(reference, zero)
        cli_out = out_dir / "local-base-reexport.crosscheck.cact"
        # The real console script, not `python -m needle.cli` - that module has
        # no __main__ guard and would exit 0 having built nothing.
        executable = _needle_executable()
        if executable is None:
            print("the `needle` console script is not on PATH; cannot cross-check", file=sys.stderr)
            return 2
        command = [executable, "build", str(checkpoint), "--lora", str(zero), "--out", str(cli_out)]
        print("  crosscheck needle build {} --lora {} --out {}".format(checkpoint, zero, cli_out))
        subprocess.run(
            command,
            check=True,
            env={**os.environ, "NEEDLE_TELEMETRY": "0", "DO_NOT_TRACK": "1"},
        )
        cli_digest = sha256_file(cli_out)
        identical = cli_digest == digest
        print(f"             cli sha256 {cli_digest}  {'IDENTICAL' if identical else 'DIFFERS'}")
        record["crossCheck"] = {
            "zeroAdapter": {
                "path": str(zero.relative_to(REPO)).replace("\\", "/"),
                "sha256": sha256_file(zero),
                **meta,
            },
            "reference": str(reference.relative_to(REPO)).replace("\\", "/"),
            "command": "needle build {} --lora {} --out {}".format(
                str(checkpoint.relative_to(REPO)).replace("\\", "/"),
                str(zero.relative_to(REPO)).replace("\\", "/"),
                cli_out.name,
            ),
            "sha256": cli_digest,
            "identicalToNoAdapterExport": identical,
        }
        cli_out.unlink()
        if not identical:
            print("the zero-adapter CLI build differs from the no-adapter export", file=sys.stderr)
            return 1

    if args.record:
        path = Path(args.record)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"  record     {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
