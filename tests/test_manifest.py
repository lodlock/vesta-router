"""Manifest validation.

CI's gate must be at least as strict as the device's. A manifest that passes
here and is refused on a phone is a release that shipped and cannot be
installed, so these cases mirror Vesta's
``apps/mobile/lib/router-model/__tests__/manifest.test.ts`` and add the two
rules this side enforces additionally (derived filename, unknown metric names).
"""

import copy
import json
import unittest
from pathlib import Path

from vesta_router.manifest import (
    SUPPORTED_MANIFEST_VERSIONS,
    sha256_file,
    validate_manifest,
    verify_artifact,
)

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "examples" / "vesta-router-1.0.0.manifest.json"

DIGEST_A = "0f1e2d3c4b5a69788796a5b4c3d2e1f00f1e2d3c4b5a69788796a5b4c3d2e1f0"
DIGEST_B = "1a2b3c4d5e6f708090a0b0c0d0e0f0011a2b3c4d5e6f708090a0b0c0d0e0f001"


def good() -> dict:
    return {
        "manifestVersion": 1,
        "modelVersion": "1.0.0",
        "status": "stable",
        "artifact": {"filename": "vesta-router-1.0.0.cact", "sha256": DIGEST_A, "sizeBytes": 35335380},
        "runtime": {
            "engine": "needle",
            "engineVersion": "3.0.0",
            "minEngineVersion": "3.0.0",
            "exporterVersion": "cactus-needle 0.4.1",
        },
        "shape": {
            "architecture": "needle3",
            "parameters": 121021910,
            "layers": 20,
            "hiddenSize": 768,
            "contextTokens": 8192,
            "quantization": "cactus-quants",
        },
        "provenance": {
            "trainingDatasetRevision": "a1b2c3d4",
            "evalDatasetRevision": "b2c3d4e5",
            "trainingConfigRevision": "c3d4e5f6",
            "parentModelVersion": None,
            "builtAt": "2026-10-01T09:14:00Z",
        },
        "compatibility": {
            "minVestaVersion": "0.2.0",
            "maxVestaVersion": None,
            "requiredRouterProtocolVersion": 1,
            "toolSchemaVersion": 2,
        },
        "evaluation": {
            "corpusRevision": "b2c3d4e5",
            "cases": 62,
            "evaluatedArtifactSha256": DIGEST_A,
            "evaluatedAt": "2026-10-01T10:02:00Z",
            "metrics": {"toolSelectionAccuracy": 0.97, "missingSlotExecutionRate": 0.0},
        },
    }


class ValidTests(unittest.TestCase):
    def test_a_good_manifest_passes(self):
        self.assertEqual(validate_manifest(good()), [])

    def test_absent_metrics_stay_absent(self):
        # Vesta's rule, and why it exists: on a false-positive-rate row, 0 reads
        # as perfect. A metric nobody measured must not read as a result.
        doc = good()
        self.assertNotIn("falsePositiveActionRate", doc["evaluation"]["metrics"])
        self.assertEqual(validate_manifest(doc), [])


class EnvelopeTests(unittest.TestCase):
    def test_unsupported_manifest_version(self):
        doc = good()
        doc["manifestVersion"] = max(SUPPORTED_MANIFEST_VERSIONS) + 1
        problems = validate_manifest(doc)
        self.assertEqual(len(problems), 1)
        self.assertIn("unsupported manifest version", problems[0])

    def test_the_version_gate_reports_alone(self):
        # A newer envelope may be perfectly correct under a schema this reader
        # has not seen. Listing its unfamiliar fields as faults reads as a broken
        # release rather than an old reader.
        problems = validate_manifest({"manifestVersion": 99, "somethingEntirelyNew": True})
        self.assertEqual(len(problems), 1)
        self.assertTrue(problems[0].startswith("manifestVersion:"))

    def test_not_an_object(self):
        self.assertNotEqual(validate_manifest([]), [])


class DigestTests(unittest.TestCase):
    def test_bad_formats_are_refused(self):
        for label, value in [
            ("too short", "0f1e2d3c"),
            ("uppercase", DIGEST_A.upper()),
            ("padded", f" {DIGEST_A} "),
            ("not hex", "g" * 64),
            ("a number", 12345),
        ]:
            with self.subTest(label):
                doc = good()
                doc["artifact"]["sha256"] = value
                self.assertTrue(any("artifact.sha256" in p for p in validate_manifest(doc)))

    def test_uppercase_is_not_normalized_into_a_pass(self):
        # Quietly lowercasing would repair a producer bug on every device that
        # reads the file, instead of surfacing it once here.
        doc = good()
        doc["artifact"]["sha256"] = DIGEST_A.upper()
        doc["evaluation"]["evaluatedArtifactSha256"] = DIGEST_A.upper()
        self.assertNotEqual(validate_manifest(doc), [])

    def test_required_at_all(self):
        doc = good()
        del doc["artifact"]["sha256"]
        self.assertTrue(any("artifact.sha256" in p for p in validate_manifest(doc)))


class MissingMetadataTests(unittest.TestCase):
    def test_top_level_sections(self):
        for key in ["modelVersion", "status", "artifact", "runtime", "shape", "provenance", "compatibility"]:
            with self.subTest(key):
                doc = good()
                del doc[key]
                self.assertTrue(any(p.startswith(key) for p in validate_manifest(doc)))

    def test_nested_fields(self):
        for section, key in [
            ("runtime", "engine"),
            ("runtime", "minEngineVersion"),
            ("shape", "contextTokens"),
            ("provenance", "trainingDatasetRevision"),
            ("provenance", "evalDatasetRevision"),
            ("provenance", "builtAt"),
            ("compatibility", "minVestaVersion"),
            ("compatibility", "requiredRouterProtocolVersion"),
            ("compatibility", "toolSchemaVersion"),
        ]:
            with self.subTest(f"{section}.{key}"):
                doc = good()
                del doc[section][key]
                self.assertTrue(any(p.startswith(f"{section}.{key}") for p in validate_manifest(doc)))

    def test_absent_nullable_is_different_from_explicit_null(self):
        # `null` is an answer - "not applicable". Absent is a producer that
        # forgot, and the two must not read the same.
        doc = good()
        del doc["provenance"]["parentModelVersion"]
        self.assertTrue(any("parentModelVersion" in p for p in validate_manifest(doc)))

    def test_every_problem_is_collected(self):
        doc = good()
        del doc["modelVersion"]
        del doc["runtime"]["engine"]
        del doc["compatibility"]["toolSchemaVersion"]
        problems = validate_manifest(doc)
        self.assertGreaterEqual(len(problems), 3)


class StatusAndVersionTests(unittest.TestCase):
    def test_stable_on_a_prerelease_is_refused(self):
        doc = good()
        doc["modelVersion"] = "1.0.0-rc.1"
        doc["artifact"]["filename"] = "vesta-router-1.0.0-rc.1.cact"
        self.assertTrue(any(p.startswith("status:") for p in validate_manifest(doc)))

    def test_candidate_on_a_release_is_refused(self):
        doc = good()
        doc["status"] = "candidate"
        self.assertTrue(any(p.startswith("status:") for p in validate_manifest(doc)))

    def test_candidate_on_an_rc_passes(self):
        doc = good()
        doc["status"] = "candidate"
        doc["modelVersion"] = "1.1.0-rc.2"
        doc["artifact"]["filename"] = "vesta-router-1.1.0-rc.2.cact"
        self.assertEqual(validate_manifest(doc), [])

    def test_unreadable_model_version(self):
        doc = good()
        doc["modelVersion"] = "v1.0"
        self.assertTrue(any(p.startswith("modelVersion:") for p in validate_manifest(doc)))

    def test_filename_must_be_the_derived_name(self):
        # Stricter than the device's validator on purpose: this is where a typo
        # can still be fixed.
        doc = good()
        doc["artifact"]["filename"] = "router.cact"
        self.assertTrue(any("derived name" in p for p in validate_manifest(doc)))


class PromotionInvariantTests(unittest.TestCase):
    def test_stable_evaluated_on_a_different_artifact_is_refused(self):
        # The failure being prevented: evaluate the adapter, promote on that
        # number, quantize afterwards, ship a .cact nobody ran.
        doc = good()
        doc["evaluation"]["evaluatedArtifactSha256"] = DIGEST_B
        self.assertTrue(
            any("must be evaluated on the artifact it ships" in p for p in validate_manifest(doc))
        )

    def test_stable_with_no_evaluation_is_refused(self):
        doc = good()
        del doc["evaluation"]
        self.assertTrue(any(p.startswith("evaluation:") for p in validate_manifest(doc)))

    def test_experimental_with_no_evaluation_is_allowed(self):
        doc = good()
        doc["status"] = "experimental"
        doc["modelVersion"] = "1.1.0-exp.3"
        doc["artifact"]["filename"] = "vesta-router-1.1.0-exp.3.cact"
        del doc["evaluation"]
        self.assertEqual(validate_manifest(doc), [])


class MetricTests(unittest.TestCase):
    def test_rates_must_be_in_range(self):
        for key, value in [("falsePositiveActionRate", 1.5), ("toolSelectionAccuracy", -0.1)]:
            with self.subTest(key):
                doc = good()
                doc["evaluation"]["metrics"][key] = value
                self.assertTrue(any(key in p for p in validate_manifest(doc)))

    def test_counts_must_be_integers(self):
        doc = good()
        doc["evaluation"]["metrics"]["staleContextMisuseCount"] = 0.5
        self.assertTrue(any("staleContextMisuseCount" in p for p in validate_manifest(doc)))

    def test_an_undefined_metric_name_is_refused(self):
        # A typo in a metric name would otherwise be silently ignored, and the
        # gate that reads it would then report NOT MEASURED for a build that
        # measured it.
        doc = good()
        doc["evaluation"]["metrics"]["toolSelectionAcuracy"] = 0.99
        self.assertTrue(any("is not a metric this contract defines" in p for p in validate_manifest(doc)))


class ExampleTests(unittest.TestCase):
    def test_the_documented_example_exists(self):
        self.assertTrue(EXAMPLE.exists())

    def test_the_documented_example_validates(self):
        problems = validate_manifest(json.loads(EXAMPLE.read_text(encoding="utf-8")))
        self.assertEqual(problems, [], "\n".join(problems))

    def test_the_documented_example_targets_the_active_tool_schema(self):
        # An example on the frozen v1 surface would teach the wrong shape to
        # everyone who copies it.
        doc = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        self.assertEqual(doc["compatibility"]["toolSchemaVersion"], 2)

    def test_the_index_example_is_consistent_with_the_manifest_example(self):
        index = json.loads((REPO / "examples" / "index.json").read_text(encoding="utf-8"))
        manifest = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        entry = next(r for r in index["releases"] if r["modelVersion"] == manifest["modelVersion"])
        self.assertEqual(entry["sha256"], manifest["artifact"]["sha256"])
        self.assertEqual(entry["sizeBytes"], manifest["artifact"]["sizeBytes"])
        self.assertEqual(entry["toolSchemaVersion"], manifest["compatibility"]["toolSchemaVersion"])

    def test_the_eval_report_example_names_the_artifact_the_manifest_ships(self):
        report = json.loads((REPO / "examples" / "eval-report-1.0.0.json").read_text(encoding="utf-8"))
        manifest = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        self.assertEqual(report["artifactSha256"], manifest["artifact"]["sha256"])
        self.assertEqual(report["artifactKind"], "cact")


class VerifyTests(unittest.TestCase):
    def test_a_manifest_that_does_not_describe_the_file_is_refused(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "vesta-router-1.0.0.cact"
            artifact.write_bytes(b"not a model")

            doc = good()
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(json.dumps(doc), encoding="utf-8")

            problems = verify_artifact(manifest_path, artifact)
            self.assertTrue(any("sizeBytes" in p for p in problems))
            self.assertTrue(any("sha256" in p for p in problems))

    def test_a_manifest_that_does_describe_the_file_passes(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "vesta-router-1.0.0.cact"
            payload = b"pretend weights"
            artifact.write_bytes(payload)

            doc = copy.deepcopy(good())
            doc["artifact"]["sizeBytes"] = len(payload)
            doc["artifact"]["sha256"] = sha256_file(artifact)
            doc["evaluation"]["evaluatedArtifactSha256"] = doc["artifact"]["sha256"]

            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(json.dumps(doc), encoding="utf-8")

            self.assertEqual(verify_artifact(manifest_path, artifact), [])

    def test_a_missing_artifact_is_reported_as_missing(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.write_text(json.dumps(good()), encoding="utf-8")
            problems = verify_artifact(manifest_path, Path(tmp) / "absent.cact")
            self.assertTrue(any("does not exist" in p for p in problems))


if __name__ == "__main__":
    unittest.main()
