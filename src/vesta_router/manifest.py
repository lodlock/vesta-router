"""Building and validating a release manifest.

The counterpart to ``apps/mobile/lib/router-model/manifest.ts`` in Vesta. That
one is the gate on the device; this one is the gate in CI, and it must be at
least as strict — a manifest that passes here and is refused on a phone is a
release that shipped and cannot be installed.

The rules that are enforced rather than expected:

* SHA-256 is required, always, and must be 64 LOWERCASE hex characters. It is
  not normalized. Quietly lowercasing an uppercase digest would repair a
  producer bug on every device that ever reads the file instead of surfacing it
  once, here, where it can be fixed.
* ``status`` and ``modelVersion`` must agree (``stable`` has no prerelease,
  ``candidate`` is an ``rc``, ``experimental`` is an ``exp``).
* A ``stable`` manifest needs an ``evaluation``, and its
  ``evaluatedArtifactSha256`` must equal the artifact's own digest. This blocks
  one specific failure: evaluate the adapter, promote on that number, quantize
  afterwards, and ship a ``.cact`` nobody ran.
* Absent metrics stay absent. A metric the harness did not produce is not 0 —
  on a false-positive-rate row, 0 reads as perfect.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from .version import STATUSES, artifact_filename, parse_model_version, status_matches_version

__all__ = [
    "SUPPORTED_MANIFEST_VERSIONS",
    "validate_manifest",
    "sha256_file",
    "verify_artifact",
    "RATE_METRICS",
    "COUNT_METRICS",
    "MEASUREMENT_METRICS",
]

SUPPORTED_MANIFEST_VERSIONS = (1,)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_APP_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")
_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$")

RATE_METRICS = (
    "toolSelectionAccuracy",
    "argumentExactMatchAccuracy",
    "falsePositiveActionRate",
    "missingSlotExecutionRate",
    "noToolClassificationAccuracy",
    "unitNormalizationAccuracy",
)
COUNT_METRICS = ("malformedOutputCount", "staleContextMisuseCount")
MEASUREMENT_METRICS = ("medianLatencyMs", "p95LatencyMs", "peakMemoryMb")


@dataclass(frozen=True)
class _Problem:
    field: str
    reason: str

    def __str__(self) -> str:
        return f"{self.field}: {self.reason}" if self.field else self.reason


class _Validator:
    def __init__(self) -> None:
        self.problems: list[_Problem] = []

    def fail(self, field: str, reason: str) -> None:
        self.problems.append(_Problem(field, reason))

    def obj(self, value: object, field: str) -> dict | None:
        if not isinstance(value, dict):
            self.fail(field, "required" if value is None else "must be an object")
            return None
        return value

    def text(self, source: dict | None, key: str, field: str) -> str | None:
        value = (source or {}).get(key)
        if not isinstance(value, str) or not value.strip():
            self.fail(field, "required" if value is None else "must be a non-empty string")
            return None
        return value

    def nullable_text(self, source: dict | None, key: str, field: str) -> str | None:
        if source is None or key not in source:
            self.fail(field, "required (use null when not applicable)")
            return None
        value = source[key]
        if value is None:
            return None
        if isinstance(value, str) and value.strip():
            return value
        self.fail(field, "must be a string or null")
        return None

    def integer(self, source: dict | None, key: str, field: str, minimum: int | None = None) -> int | None:
        value = (source or {}).get(key)
        if not isinstance(value, int) or isinstance(value, bool):
            self.fail(field, "required" if value is None else "must be an integer")
            return None
        if minimum is not None and value < minimum:
            self.fail(field, f"must be at least {minimum}")
            return None
        return value

    def digest(self, source: dict | None, key: str, field: str) -> str | None:
        value = (source or {}).get(key)
        if not isinstance(value, str):
            self.fail(field, "required" if value is None else "must be a string")
            return None
        if not _SHA256_RE.match(value):
            self.fail(field, "must be 64 lowercase hexadecimal characters")
            return None
        return value

    def timestamp(self, source: dict | None, key: str, field: str) -> str | None:
        value = self.text(source, key, field)
        if value is None:
            return None
        if not _ISO_RE.match(value):
            self.fail(field, "must be an ISO-8601 timestamp")
            return None
        return value

    def app_version(self, source: dict | None, key: str, field: str) -> str | None:
        value = self.text(source, key, field)
        if value is None:
            return None
        if not _APP_VERSION_RE.match(value):
            self.fail(field, "must be a MAJOR.MINOR.PATCH app version")
            return None
        return value


def _validate_metrics(v: _Validator, raw: object) -> None:
    metrics = v.obj(raw, "evaluation.metrics")
    if metrics is None:
        return
    for key in RATE_METRICS:
        if key not in metrics:
            continue
        value = metrics[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= 1:
            v.fail(f"evaluation.metrics.{key}", "must be a number in [0,1]")
    for key in COUNT_METRICS:
        if key not in metrics:
            continue
        value = metrics[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            v.fail(f"evaluation.metrics.{key}", "must be a non-negative integer")
    for key in MEASUREMENT_METRICS:
        if key not in metrics:
            continue
        value = metrics[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            v.fail(f"evaluation.metrics.{key}", "must be a non-negative number")
    for key in metrics:
        if key not in RATE_METRICS + COUNT_METRICS + MEASUREMENT_METRICS:
            v.fail(f"evaluation.metrics.{key}", "is not a metric this contract defines")


def validate_manifest(doc: object) -> list[str]:
    """Return a list of problems. An empty list means the manifest is valid."""
    v = _Validator()

    root = v.obj(doc, "")
    if root is None:
        return [str(p) for p in v.problems]

    manifest_version = v.integer(root, "manifestVersion", "manifestVersion", minimum=1)
    if manifest_version is None:
        return [str(p) for p in v.problems]
    if manifest_version not in SUPPORTED_MANIFEST_VERSIONS:
        # Reported alone. A newer envelope may be perfectly correct under a
        # schema this build has not seen, and listing its unfamiliar fields as
        # faults reads as a broken release rather than an old reader.
        return [
            f"manifestVersion: unsupported manifest version {manifest_version}; "
            f"this build reads {', '.join(str(x) for x in SUPPORTED_MANIFEST_VERSIONS)}"
        ]

    model_version_raw = v.text(root, "modelVersion", "modelVersion")
    model_version = parse_model_version(model_version_raw) if model_version_raw else None
    if model_version_raw and model_version is None:
        v.fail("modelVersion", "must be MAJOR.MINOR.PATCH with an optional -rc.N or -exp.N suffix")

    status = root.get("status")
    if status not in STATUSES:
        v.fail("status", f"must be one of {', '.join(STATUSES)}")
        status = None
    elif model_version and not status_matches_version(status, model_version):
        v.fail("status", f'status "{status}" disagrees with version "{model_version.raw}"')

    artifact = v.obj(root.get("artifact"), "artifact")
    filename = v.text(artifact, "filename", "artifact.filename")
    sha256 = v.digest(artifact, "sha256", "artifact.sha256")
    v.integer(artifact, "sizeBytes", "artifact.sizeBytes", minimum=1)
    if filename and not filename.endswith(".cact"):
        v.fail("artifact.filename", "must be a .cact file")
    if filename and model_version and filename != artifact_filename(model_version.raw):
        v.fail(
            "artifact.filename",
            f"must be the derived name {artifact_filename(model_version.raw)}",
        )

    runtime = v.obj(root.get("runtime"), "runtime")
    v.text(runtime, "engine", "runtime.engine")
    v.text(runtime, "engineVersion", "runtime.engineVersion")
    v.text(runtime, "minEngineVersion", "runtime.minEngineVersion")
    v.nullable_text(runtime, "exporterVersion", "runtime.exporterVersion")

    shape = v.obj(root.get("shape"), "shape")
    v.text(shape, "architecture", "shape.architecture")
    v.integer(shape, "parameters", "shape.parameters", minimum=1)
    v.integer(shape, "layers", "shape.layers", minimum=1)
    v.integer(shape, "contextTokens", "shape.contextTokens", minimum=1)
    v.text(shape, "quantization", "shape.quantization")

    provenance = v.obj(root.get("provenance"), "provenance")
    v.text(provenance, "trainingDatasetRevision", "provenance.trainingDatasetRevision")
    v.text(provenance, "evalDatasetRevision", "provenance.evalDatasetRevision")
    v.text(provenance, "trainingConfigRevision", "provenance.trainingConfigRevision")
    parent = v.nullable_text(provenance, "parentModelVersion", "provenance.parentModelVersion")
    if parent is not None and parse_model_version(parent) is None:
        v.fail("provenance.parentModelVersion", "must be a router model version or null")
    v.timestamp(provenance, "builtAt", "provenance.builtAt")

    compat = v.obj(root.get("compatibility"), "compatibility")
    v.app_version(compat, "minVestaVersion", "compatibility.minVestaVersion")
    if compat is not None and compat.get("maxVestaVersion") is not None:
        v.app_version(compat, "maxVestaVersion", "compatibility.maxVestaVersion")
    elif compat is not None and "maxVestaVersion" not in compat:
        v.fail("compatibility.maxVestaVersion", "required (use null when not applicable)")
    v.integer(
        compat, "requiredRouterProtocolVersion", "compatibility.requiredRouterProtocolVersion", minimum=1
    )
    v.integer(compat, "toolSchemaVersion", "compatibility.toolSchemaVersion", minimum=1)

    evaluation = root.get("evaluation")
    if evaluation is not None:
        section = v.obj(evaluation, "evaluation")
        v.text(section, "corpusRevision", "evaluation.corpusRevision")
        v.integer(section, "cases", "evaluation.cases", minimum=1)
        evaluated_digest = v.digest(section, "evaluatedArtifactSha256", "evaluation.evaluatedArtifactSha256")
        v.timestamp(section, "evaluatedAt", "evaluation.evaluatedAt")
        _validate_metrics(v, (section or {}).get("metrics"))
    else:
        evaluated_digest = None

    if status == "stable":
        if evaluation is None:
            v.fail("evaluation", "required for a stable release")
        elif sha256 and evaluated_digest and evaluated_digest != sha256:
            v.fail(
                "evaluation.evaluatedArtifactSha256",
                "a stable release must be evaluated on the artifact it ships; "
                "this digest is a different file",
            )

    notes = root.get("releaseNotes")
    if notes is not None and not isinstance(notes, str):
        v.fail("releaseNotes", "must be a string when present")

    return [str(p) for p in v.problems]


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_artifact(manifest_path: str | Path, artifact_path: str | Path) -> list[str]:
    """Check a manifest against the bytes it describes.

    Separate from ``validate_manifest`` because they answer different questions
    and fail at different times: one is about a document, the other about a
    file that may not exist yet when the document is written.
    """
    problems: list[str] = []
    doc = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    problems.extend(validate_manifest(doc))
    if problems:
        return problems

    artifact = Path(artifact_path)
    if not artifact.exists():
        return [f"artifact: {artifact} does not exist"]

    declared = doc["artifact"]
    actual_size = artifact.stat().st_size
    if actual_size != declared["sizeBytes"]:
        problems.append(f"artifact.sizeBytes: manifest says {declared['sizeBytes']}, file is {actual_size}")

    actual_digest = sha256_file(artifact)
    if actual_digest != declared["sha256"]:
        problems.append(
            f"artifact.sha256: manifest says {declared['sha256']}, file hashes to {actual_digest}"
        )

    return problems
