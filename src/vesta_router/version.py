"""The router model version scheme.

A deliberate mirror of ``apps/mobile/lib/router-model/version.ts`` in the Vesta
repository. Two implementations of one rule is a cost, and the alternative is
worse: the app must decide "is this newer, and is it stable" with no Python
available, and the release pipeline must decide the same thing with no
TypeScript. Keeping them in step is what ``tests/test_version.py`` and Vesta's
``version.test.ts`` are for — they assert the same ordering chain, case for
case.

The scheme::

    MAJOR.MINOR.PATCH[-rc.N | -exp.N]

    MAJOR   the tool surface or output contract changed incompatibly
    MINOR   retrained: new data, new coverage, better behaviour
    PATCH   the same training, rebuilt (requantized, re-exported)
    -rc.N   a promotion candidate for the release it names
    -exp.N  a throwaway experiment, never promoted

Ordering, which is semver's::

    1.2.0-exp.1 < 1.2.0-rc.1 < 1.2.0-rc.2 < 1.2.0 < 1.2.1 < 1.3.0

Status and version agree by construction, and it is validated: ``stable`` has no
prerelease, ``candidate`` is an ``rc``, ``experimental`` is an ``exp``. So
"newer or older" and "stable or not" are both answerable from the version string
alone, which is the point — the alternative is reading it off a filename, and a
filename is whatever somebody typed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "ModelVersion",
    "parse_model_version",
    "artifact_filename",
    "status_implied_by_version",
    "status_matches_version",
    "STATUSES",
    "TERMINAL_STATUSES",
]

_VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:-(rc|exp)\.(\d+))?$")

STATUSES = ("experimental", "candidate", "stable", "rejected", "superseded")

#: Applied to a finished build after the fact, so they constrain no version shape.
TERMINAL_STATUSES = ("rejected", "superseded")

#: Release sorts above any prerelease; rc above exp.
_PRERELEASE_RANK = {None: 2, "rc": 1, "exp": 0}


@dataclass(frozen=True, order=False)
class ModelVersion:
    major: int
    minor: int
    patch: int
    prerelease_tag: str | None
    prerelease_ordinal: int | None
    raw: str

    def _key(self) -> tuple[int, int, int, int, int]:
        return (
            self.major,
            self.minor,
            self.patch,
            _PRERELEASE_RANK[self.prerelease_tag],
            self.prerelease_ordinal or 0,
        )

    def __lt__(self, other: ModelVersion) -> bool:
        return self._key() < other._key()

    def __le__(self, other: ModelVersion) -> bool:
        return self._key() <= other._key()

    def __gt__(self, other: ModelVersion) -> bool:
        return self._key() > other._key()

    def __ge__(self, other: ModelVersion) -> bool:
        return self._key() >= other._key()


def parse_model_version(raw: object) -> ModelVersion | None:
    """Parse a version, or return ``None``.

    Returns ``None`` rather than raising for anything that does not match
    exactly — a leading ``v``, a build suffix, a prerelease tag the scheme does
    not define. Fail closed: an unparseable version is not "probably newer", and
    the only alternative to refusing it is ordering it by guesswork.
    """
    if not isinstance(raw, str):
        return None
    match = _VERSION_RE.match(raw)
    if not match:
        return None
    major, minor, patch, tag, ordinal = match.groups()
    return ModelVersion(
        major=int(major),
        minor=int(minor),
        patch=int(patch),
        prerelease_tag=tag,
        prerelease_ordinal=int(ordinal) if ordinal is not None else None,
        raw=raw,
    )


def status_implied_by_version(version: ModelVersion) -> str:
    if version.prerelease_tag is None:
        return "stable"
    return "candidate" if version.prerelease_tag == "rc" else "experimental"


def status_matches_version(status: str, version: ModelVersion) -> bool:
    """Whether a declared status is allowed to sit on this version string."""
    if status in TERMINAL_STATUSES:
        return True
    return status == status_implied_by_version(version)


def artifact_filename(model_version: str) -> str:
    """The canonical artifact name.

    Derived, never authoritative — the manifest is. This exists so the
    derivation happens in one place and a release cannot quietly ship the
    spike's permanently ambiguous ``needle3.cact`` again.
    """
    return f"vesta-router-{model_version}.cact"
