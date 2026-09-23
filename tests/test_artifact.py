"""Reading a ``.cact`` header and tensor directory, and decomposing a size.

These tests build real archives in memory — a valid header, a codebook, a
directory of records and 64-byte-aligned blobs — rather than loading a 35 MB
file. That keeps them in the fast tier, and it lets a case be constructed that
no artifact on disk exhibits: the mixed-width archive whose existence was the
whole finding is easy to build and hard to obtain.

The test that matters most is ``unexplainedBytes == 0``. The decomposition is
only worth reading if every byte of a difference is attributed to a named
cause, and a residual is how this module says it does not understand the format
as well as it claims.
"""

import struct
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from vesta_router.artifact import (
    CACT_TAG,
    compare_artifacts,
    inspect_artifact,
    read_directory,
    read_header,
    render_text,
)

_HEADER_FMT = "<48If"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_RECORD_FMT = "<BBHIIIIQQII"
_RECORD_SIZE = struct.calcsize(_RECORD_FMT)
FP16, FP32, CQ, RAW = 1, 2, 3, 4


def build(tensors, *, num_layers=2, d_model=8, codebook=(0.25, 0.5), tag=CACT_TAG) -> bytes:
    """A structurally valid archive.

    ``tensors`` is ``(dtype, shape, nbytes, bits)``, optionally with a fifth
    element of real payload bytes. Blobs are otherwise zero-filled: the only
    tensor whose CONTENTS are read back is ``heads.manifest``, and it is the
    one place a test has to supply them.
    """
    header = [0] * 48
    header[0] = tag
    header[1] = len(tensors)
    header[2] = len(codebook)
    header[3] = 256  # kv_window
    header[4] = 8  # kv_bits
    header[5] = 64  # vocab
    header[6] = 64  # out_vocab
    header[7] = d_model
    header[8] = 2  # num_heads
    header[9] = 1  # num_kv_heads
    header[10] = num_layers
    header[26] = 2  # num engram orders
    header[27], header[28] = 2, 3
    header[31] = 1  # num engram sites

    preamble = struct.pack(_HEADER_FMT, *header, 10000.0) + b"".join(
        struct.pack("<f", value) for value in codebook
    )
    directory_size = len(tensors) * _RECORD_SIZE
    offset = len(preamble) + directory_size
    records, blobs = [], []
    for tensor in tensors:
        dtype, shape, nbytes, bits = tensor[:4]
        payload = tensor[4] if len(tensor) > 4 else bytes(nbytes)
        offset = (offset + 63) & ~63  # the format's 64-byte alignment
        padded = shape + (0,) * (4 - len(shape))
        records.append(struct.pack(_RECORD_FMT, dtype, len(shape), 0, *padded, offset, nbytes, 128, bits))
        blobs.append((offset, payload.ljust(nbytes, b"\x00")))
        offset += nbytes

    buffer = bytearray(offset)
    buffer[: len(preamble)] = preamble
    position = len(preamble)
    for record in records:
        buffer[position : position + _RECORD_SIZE] = record
        position += _RECORD_SIZE
    for start, blob in blobs:
        buffer[start : start + len(blob)] = blob
    return bytes(buffer)


def write(directory, name, payload) -> Path:
    path = Path(directory) / name
    path.write_bytes(payload)
    return path


#: A body of two matmul tensors plus a norm and a tokenizer, at a given width.
def body(bits, cq_bytes):
    return [
        (CQ, (64, 8), cq_bytes, bits),
        (CQ, (8, 8), cq_bytes, bits),
        (FP16, (8,), 16, 0),
        (RAW, (), 24, 0),
    ]


#: `heads.manifest` (one code) plus one probe head, in the format's own layout.
#: `gain` is FP16 2-D with first dim num_layers + 1, which is what names it.
def confidence_head(num_layers=2):
    rows = num_layers + 1
    return [
        # heads.manifest: one FP16 code. 2 is ConfidenceHead.code, and reading
        # it is what NAMES the head — a head identified from its shape would be
        # a guess, since embedding and confidence differ only in one dimension.
        (FP16, (1,), 2, 0, struct.pack("<e", 2.0)),
        (CQ, (rows * 4, 8), 64, 4),  # probes
        (FP16, (rows, 4), 24, 0),  # gain
        (CQ, (4, 8), 16, 4),  # query
        (FP16, (4, rows, 4), 96, 0),  # row_bias
        (CQ, (1, 32), 16, 4),  # proj
        (FP16, (1,), 2, 0),  # bias
    ]


class ReadingAHeader(unittest.TestCase):
    def test_a_foreign_file_is_refused_rather_than_parsed(self):
        with self.assertRaises(ValueError) as raised:
            read_header(build(body(4, 64), tag=0xDEADBEEF), "x.cact")
        self.assertIn("not a Needle 3", str(raised.exception))

    def test_a_truncated_file_is_refused(self):
        with self.assertRaises(ValueError):
            read_header(b"\x00" * 8, "x.cact")

    def test_geometry_is_read_by_name(self):
        header = read_header(build(body(4, 64), num_layers=5, d_model=16))
        self.assertEqual(header["numLayers"], 5)
        self.assertEqual(header["dModel"], 16)
        self.assertEqual(header["kvBits"], 8)
        self.assertEqual(header["numTensors"], 4)

    def test_the_directory_is_positional(self):
        raw = build(body(4, 64))
        records = read_directory(raw, read_header(raw))
        self.assertEqual([r["index"] for r in records], [0, 1, 2, 3])
        self.assertEqual(records[0]["dtype"], "CQ")
        self.assertEqual(records[-1]["dtype"], "RAW")


class InspectingAnArchive(unittest.TestCase):
    def test_cq_widths_are_reported_per_width(self):
        with TemporaryDirectory() as directory:
            path = write(directory, "a.cact", build(body(2, 32)))
            document = inspect_artifact(path)
        self.assertEqual(document["tensors"]["cqWidths"]["2"]["tensors"], 2)
        self.assertEqual(document["tensors"]["uniformCqWidth"], 2)

    def test_a_mixed_width_archive_reports_no_uniform_width(self):
        # The published base's actual shape: mostly W2 with a handful of W4.
        mixed = [(CQ, (64, 8), 64, 4), (CQ, (8, 8), 32, 2), (RAW, (), 24, 0)]
        with TemporaryDirectory() as directory:
            path = write(directory, "a.cact", build(mixed))
            document = inspect_artifact(path)
        self.assertIsNone(document["tensors"]["uniformCqWidth"])
        self.assertEqual(sorted(document["tensors"]["cqWidths"]), ["2", "4"])

    def test_a_probe_head_is_detected_and_named_from_its_manifest(self):
        tensors = body(2, 32)[:-1] + confidence_head() + [(RAW, (), 24, 0)]
        with TemporaryDirectory() as directory:
            path = write(directory, "a.cact", build(tensors))
            document = inspect_artifact(path)
        heads = document["probeHeads"]
        self.assertTrue(heads["present"])
        self.assertEqual(heads["heads"], ["confidence"])
        # manifest + six head tensors: the seven the local export drops.
        self.assertEqual(heads["tensors"], 7)

    def test_an_archive_with_no_head_says_so(self):
        with TemporaryDirectory() as directory:
            path = write(directory, "a.cact", build(body(4, 64)))
            document = inspect_artifact(path)
        self.assertFalse(document["probeHeads"]["present"])
        self.assertEqual(document["probeHeads"]["heads"], [])

    def test_the_tokenizer_blob_is_digested_separately(self):
        with TemporaryDirectory() as directory:
            path = write(directory, "a.cact", build(body(4, 64)))
            document = inspect_artifact(path)
        self.assertEqual(document["tokenizer"]["sizeBytes"], 24)
        self.assertEqual(len(document["tokenizer"]["sha256"]), 64)

    def test_the_layout_accounts_for_every_byte(self):
        with TemporaryDirectory() as directory:
            path = write(directory, "a.cact", build(body(4, 64)))
            document = inspect_artifact(path)
        layout = document["layout"]
        total = (
            layout["headerBytes"]
            + layout["codebookBytes"]
            + layout["directoryBytes"]
            + layout["tensorBytes"]
            + layout["alignmentPaddingBytes"]
        )
        self.assertEqual(total, document["sizeBytes"])

    def test_render_text_names_the_widths(self):
        with TemporaryDirectory() as directory:
            path = write(directory, "a.cact", build(body(2, 32)))
            text = render_text(inspect_artifact(path))
        self.assertIn("CQ W2", text)
        self.assertIn("probe heads: none", text)


class DecomposingASizeDifference(unittest.TestCase):
    """The published-versus-local question, in miniature."""

    def _pair(self, directory):
        published = body(2, 32)[:-1] + confidence_head() + [(RAW, (), 24, 0)]
        local = body(4, 64)
        return (
            write(directory, "published.cact", build(published)),
            write(directory, "local.cact", build(local)),
        )

    def test_every_byte_of_the_difference_is_attributed(self):
        with TemporaryDirectory() as directory:
            before, after = self._pair(directory)
            document = compare_artifacts(before, after)
        self.assertEqual(document["sizeDecomposition"]["unexplainedBytes"], 0)

    def test_the_precision_term_carries_the_difference(self):
        with TemporaryDirectory() as directory:
            before, after = self._pair(directory)
            document = compare_artifacts(before, after)
        decomposition = document["sizeDecomposition"]
        self.assertGreater(decomposition["modelBodyTensorBytes"], 0)
        # The head is only removed, so its term is negative — it cannot be the
        # cause of a file getting bigger.
        self.assertLess(decomposition["probeHeadBytes"], 0)

    def test_the_width_table_shows_the_map_changing(self):
        with TemporaryDirectory() as directory:
            before, after = self._pair(directory)
            document = compare_artifacts(before, after)
        widths = {row["bits"]: row for row in document["cqWidthsByBits"]}
        self.assertEqual(widths[2]["afterTensors"], 0)
        self.assertEqual(widths[4]["beforeTensors"], 3)  # the head's three
        self.assertGreater(widths[4]["afterTensors"], 0)

    def test_the_same_geometry_is_recognised(self):
        with TemporaryDirectory() as directory:
            before, after = self._pair(directory)
            document = compare_artifacts(before, after)
        self.assertTrue(document["sameGeometry"])
        self.assertTrue(document["sameTokenizer"])

    def test_positions_that_changed_width_are_counted(self):
        with TemporaryDirectory() as directory:
            before = write(directory, "a.cact", build(body(2, 32)))
            after = write(directory, "b.cact", build(body(4, 64)))
            document = compare_artifacts(before, after)
        self.assertEqual(document["positionsDiffering"], 2)
        self.assertEqual(document["positionDeltas"][0]["before"]["bits"], 2)
        self.assertEqual(document["positionDeltas"][0]["after"]["bits"], 4)

    def test_an_archive_compared_with_itself_decomposes_to_nothing(self):
        with TemporaryDirectory() as directory:
            path = write(directory, "a.cact", build(body(4, 64)))
            document = compare_artifacts(path, path)
        for value in document["sizeDecomposition"].values():
            self.assertEqual(value, 0)
        self.assertEqual(document["positionsDiffering"], 0)


if __name__ == "__main__":
    unittest.main()
