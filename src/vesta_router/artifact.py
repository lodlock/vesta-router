"""Reading a ``.cact`` archive's header and tensor directory, and diffing two.

WHY THIS EXISTS

candidate-r1's build record carried an unexplained finding: the locally built
``.cact`` is 63.4 MB where the published base archive is 35.3 MB, for the same
20-layer architecture. "Nominally the same quantization" was an assumption, and
an assumption about numerics sitting underneath a behavioural comparison is the
kind that makes the comparison mean something other than what it says.

It is answerable from the file. A ``.cact`` is a fixed header, a shared
codebook, a nameless tensor directory and 64-byte-aligned blobs — and the
directory records each tensor's dtype, shape, byte count, quantization group
and **bit width**. Reading those answers "what is actually in here" without
loading a runtime, without numpy and without dequantizing anything.

WHAT IT DELIBERATELY DOES NOT DO

No tensor payload is decoded except the probe-head manifest, which is a handful
of FP16 codes and is what names a head rather than guessing from its shape. So
this module cannot say whether two archives hold the *same weights*; it says
what shape, dtype and precision those weights are stored at. That is the
question the size discrepancy actually posed.

THE FORMAT IS READ FROM ``needle.model.export``'s own specification and the
constants are restated here rather than imported, for the same reason the rest
of this package restates Vesta's rules instead of importing them: the fast tier
must run with nothing installed, and a drift between the two is a finding worth
having rather than an error to avoid. :func:`inspect_artifact` refuses a file
whose tag is not Needle 3's, so a format change surfaces as a refusal.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

__all__ = [
    "CACT_TAG",
    "compare_artifacts",
    "inspect_artifact",
    "read_directory",
    "read_header",
]

#: ``needle.model.export.TAG``. A file that does not start with this is not a
#: Needle 3 archive and is refused rather than parsed into nonsense.
CACT_TAG = 0x05E12A84

_HEADER_FMT = "<48If"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_RECORD_FMT = "<BBHIIIIQQII"
_RECORD_SIZE = struct.calcsize(_RECORD_FMT)
_ALIGN = 64

#: ``needle.model.export``'s dtype codes.
DTYPES = {1: "FP16", 2: "FP32", 3: "CQ", 4: "RAW"}

#: ``needle.model.architecture.HEADS`` codes, as the ``heads.manifest`` tensor
#: carries them. A head is named from this rather than inferred from its shape.
HEAD_CODES = {1: "embedding", 2: "confidence", 3: "router"}

#: The header's u32 fields after the five fixed ones, in declaration order.
_GEOMETRY_FIELDS = (
    "vocab",
    "outVocab",
    "dModel",
    "numHeads",
    "numKvHeads",
    "numLayers",
    "qkHeadDim",
    "vHeadDim",
    "maxSeqLen",
    "hadaN",
    "mhcLanes",
    "slidingWindow",
)


def read_header(raw: bytes, path: str | Path = "<bytes>") -> dict:
    """The fixed header, as named fields. Raises on a foreign file."""
    if len(raw) < _HEADER_SIZE:
        raise ValueError(f"{path}: too short to hold a .cact header")
    fields = struct.unpack_from(_HEADER_FMT, raw, 0)
    if fields[0] != CACT_TAG:
        raise ValueError(f"{path}: not a Needle 3 .cact archive (tag {fields[0]:#x})")

    header = {
        "tag": fields[0],
        "numTensors": fields[1],
        "codebookLen": fields[2],
        "kvWindow": fields[3],
        "kvBits": fields[4],
    }
    header.update(dict(zip(_GEOMETRY_FIELDS, fields[5:17], strict=True)))
    header["globalLayers"] = tuple(
        i for i in range(header["numLayers"]) if (fields[17] | (fields[18] << 32)) >> i & 1
    )
    header["qkvConvTaps"] = fields[19]
    header["engramSlots"] = fields[20]
    header["engramSubDim"] = fields[21]
    header["numEngramTables"] = fields[22]
    header["engramOrders"] = tuple(fields[27 : 27 + min(fields[26], 4)])
    header["engramLayers"] = tuple(fields[32 : 32 + min(fields[31], 16)])
    header["ropeTheta"] = fields[48]
    return header


def read_directory(raw: bytes, header: dict) -> list[dict]:
    """Every directory record, in the archive's own positional order.

    Positional because the directory is nameless: the runtime indexes tensors
    by position in a fixed canonical order, so position IS the identity and two
    archives are comparable index by index.
    """
    offset = _HEADER_SIZE + header["codebookLen"] * 4
    records = []
    for index in range(header["numTensors"]):
        fields = struct.unpack_from(_RECORD_FMT, raw, offset)
        offset += _RECORD_SIZE
        records.append(
            {
                "index": index,
                "dtype": DTYPES.get(fields[0], str(fields[0])),
                "shape": list(fields[3 : 3 + fields[1]]),
                "offset": fields[7],
                "sizeBytes": fields[8],
                "groupSize": fields[9],
                "bits": fields[10],
            }
        )
    return records


def _probe_heads(raw: bytes, records: list[dict], num_layers: int) -> dict:
    """The probe heads appended after ``final_norm``, named from the manifest.

    The marker is unambiguous and comes from the format's own specification: a
    head's ``gain`` is the only FP16 2-D tensor whose first dimension is
    ``num_layers + 1`` (rows are the embedding plus every block). The
    ``heads.manifest`` immediately before the first head is a 1-D FP16 tensor
    of one code per head, and the codes are what name them - a head identified
    by its shape would be a guess, and ``embedding`` and ``confidence`` differ
    only in a trailing dimension.
    """
    rows = num_layers + 1
    gains = [r for r in records if r["dtype"] == "FP16" and len(r["shape"]) == 2 and r["shape"][0] == rows]
    if not gains:
        return {"present": False, "count": 0, "heads": [], "tensors": 0, "sizeBytes": 0}

    first = min(r["index"] for r in gains)
    manifest = None
    for record in reversed(records[:first]):
        if record["dtype"] == "FP16" and len(record["shape"]) == 1 and record["shape"][0] == len(gains):
            manifest = record
            break

    names: list[str] = []
    if manifest is not None:
        blob = raw[manifest["offset"] : manifest["offset"] + manifest["sizeBytes"]]
        for code in struct.unpack(f"<{len(gains)}e", blob):
            names.append(HEAD_CODES.get(int(round(code)), f"code {int(round(code))}"))

    start = manifest["index"] if manifest is not None else first
    block = [r for r in records[start:] if r["dtype"] != "RAW"]
    return {
        "present": True,
        "count": len(gains),
        "heads": names,
        "manifestIndex": manifest["index"] if manifest is not None else None,
        "tensors": len(block),
        "sizeBytes": sum(r["sizeBytes"] for r in block),
    }


def inspect_artifact(path: str | Path) -> dict:
    """Everything the header and directory of one ``.cact`` say about it."""
    path = Path(path)
    raw = path.read_bytes()
    header = read_header(raw, path)
    records = read_directory(raw, header)

    by_dtype: dict[str, dict[str, int]] = {}
    cq_widths: dict[str, dict[str, int]] = {}
    for record in records:
        entry = by_dtype.setdefault(record["dtype"], {"tensors": 0, "sizeBytes": 0})
        entry["tensors"] += 1
        entry["sizeBytes"] += record["sizeBytes"]
        if record["dtype"] == "CQ":
            width = cq_widths.setdefault(str(record["bits"]), {"tensors": 0, "sizeBytes": 0})
            width["tensors"] += 1
            width["sizeBytes"] += record["sizeBytes"]

    tokenizer = next((r for r in records if r["dtype"] == "RAW"), None)
    tensor_bytes = sum(r["sizeBytes"] for r in records)
    directory_bytes = len(records) * _RECORD_SIZE
    preamble = _HEADER_SIZE + header["codebookLen"] * 4

    return {
        "inspectorVersion": 1,
        "path": str(path),
        "sizeBytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "header": {
            key: (list(value) if isinstance(value, tuple) else value) for key, value in header.items()
        },
        "tensors": {
            "count": len(records),
            "byDtype": dict(sorted(by_dtype.items())),
            # The one number the size question turns on. CQ is the only dtype
            # whose record carries a width, and a mixed map is visible here and
            # nowhere else in the file.
            "cqWidths": dict(sorted(cq_widths.items(), key=lambda item: int(item[0]))),
            "uniformCqWidth": (int(next(iter(cq_widths))) if len(cq_widths) == 1 else None),
        },
        "probeHeads": _probe_heads(raw, records, header["numLayers"]),
        "tokenizer": (
            None
            if tokenizer is None
            else {
                "sizeBytes": tokenizer["sizeBytes"],
                "sha256": hashlib.sha256(
                    raw[tokenizer["offset"] : tokenizer["offset"] + tokenizer["sizeBytes"]]
                ).hexdigest(),
            }
        ),
        "layout": {
            "headerBytes": _HEADER_SIZE,
            "codebookBytes": header["codebookLen"] * 4,
            "directoryBytes": directory_bytes,
            "tensorBytes": tensor_bytes,
            # What the 64-byte alignment costs. Reported so the decomposition
            # below adds up exactly rather than nearly.
            "alignmentPaddingBytes": len(raw) - preamble - directory_bytes - tensor_bytes,
        },
    }


def _position_deltas(left: list[dict], right: list[dict]) -> list[dict]:
    """Positions where two directories disagree about dtype, shape or width."""

    def summarize(record: dict) -> dict:
        return {key: record[key] for key in ("dtype", "shape", "bits", "sizeBytes")}

    deltas = []
    for a, b in zip(left, right, strict=False):
        if (a["dtype"], a["shape"], a["bits"]) == (b["dtype"], b["shape"], b["bits"]):
            continue
        deltas.append({"index": a["index"], "before": summarize(a), "after": summarize(b)})
    return deltas


def compare_artifacts(before: str | Path, after: str | Path) -> dict:
    """Diff two archives structurally, and decompose the size difference.

    The decomposition is exact by construction: every byte of the difference is
    attributed to a named cause, and ``unexplainedBytes`` is the residual. A
    non-zero residual means this module does not understand the format as well
    as it claims, and saying so is the point of computing it.
    """
    left, right = inspect_artifact(before), inspect_artifact(after)
    raw_left = Path(before).read_bytes()
    raw_right = Path(after).read_bytes()
    records_left = read_directory(raw_left, read_header(raw_left, before))
    records_right = read_directory(raw_right, read_header(raw_right, after))

    widths = sorted({*left["tensors"]["cqWidths"], *right["tensors"]["cqWidths"]}, key=int)
    cq_by_width = []
    for width in widths:
        a = left["tensors"]["cqWidths"].get(width, {"tensors": 0, "sizeBytes": 0})
        b = right["tensors"]["cqWidths"].get(width, {"tensors": 0, "sizeBytes": 0})
        cq_by_width.append(
            {
                "bits": int(width),
                "beforeTensors": a["tensors"],
                "afterTensors": b["tensors"],
                "beforeBytes": a["sizeBytes"],
                "afterBytes": b["sizeBytes"],
                "deltaBytes": b["sizeBytes"] - a["sizeBytes"],
            }
        )

    head_delta = right["probeHeads"]["sizeBytes"] - left["probeHeads"]["sizeBytes"]
    directory_delta = right["layout"]["directoryBytes"] - left["layout"]["directoryBytes"]
    padding_delta = right["layout"]["alignmentPaddingBytes"] - left["layout"]["alignmentPaddingBytes"]
    codebook_delta = right["layout"]["codebookBytes"] - left["layout"]["codebookBytes"]

    # Weight bytes with the probe heads taken out of both sides, so the
    # precision term is about the model body and not about a head that one
    # archive carries and the other does not.
    body_before = left["layout"]["tensorBytes"] - left["probeHeads"]["sizeBytes"]
    body_after = right["layout"]["tensorBytes"] - right["probeHeads"]["sizeBytes"]
    body_delta = body_after - body_before

    total = right["sizeBytes"] - left["sizeBytes"]
    accounted = body_delta + head_delta + directory_delta + padding_delta + codebook_delta

    return {
        "comparisonVersion": 1,
        "before": left,
        "after": right,
        "sameGeometry": left["header"] == right["header"]
        or {k: v for k, v in left["header"].items() if k != "numTensors"}
        == {k: v for k, v in right["header"].items() if k != "numTensors"},
        "sameTokenizer": left["tokenizer"] == right["tokenizer"],
        "cqWidthsByBits": cq_by_width,
        "positionsDiffering": len(_position_deltas(records_left, records_right)),
        "positionDeltas": _position_deltas(records_left, records_right)[:20],
        "sizeDecomposition": {
            "totalDeltaBytes": total,
            "modelBodyTensorBytes": body_delta,
            "probeHeadBytes": head_delta,
            "tensorDirectoryBytes": directory_delta,
            "codebookBytes": codebook_delta,
            "alignmentPaddingBytes": padding_delta,
            "unexplainedBytes": total - accounted,
        },
    }


def render_text(document: dict) -> str:
    """A short human summary of one inspection. The JSON stays authoritative."""
    lines = [
        f"{document['path']}",
        f"  {document['sizeBytes']} bytes  sha256 {document['sha256']}",
        f"  {document['header']['numLayers']} layers  d_model {document['header']['dModel']}  "
        f"vocab {document['header']['vocab']}  kv_bits {document['header']['kvBits']}",
        f"  {document['tensors']['count']} tensors",
    ]
    for dtype, entry in document["tensors"]["byDtype"].items():
        lines.append(f"    {dtype:<5} {entry['tensors']:>4} tensors  {entry['sizeBytes']:>11} bytes")
    for bits, entry in document["tensors"]["cqWidths"].items():
        lines.append(f"    CQ W{bits} {entry['tensors']:>4} tensors  {entry['sizeBytes']:>11} bytes")
    heads = document["probeHeads"]
    lines.append(
        f"  probe heads: {', '.join(heads['heads']) if heads['heads'] else 'none'}"
        + (f"  ({heads['tensors']} tensors, {heads['sizeBytes']} bytes)" if heads["present"] else "")
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin CLI shim
    import argparse

    parser = argparse.ArgumentParser(description="inspect a .cact archive")
    parser.add_argument("artifact", nargs="+")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if len(args.artifact) == 2 and args.json:
        print(json.dumps(compare_artifacts(*args.artifact), indent=2, sort_keys=True))
        return 0
    for path in args.artifact:
        document = inspect_artifact(path)
        print(json.dumps(document, indent=2, sort_keys=True) if args.json else render_text(document))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
