"""The Needle engine adapter — the only module here that touches a runtime.

WHAT IT EVALUATES

The deployable ``.cact``, through Needle's own native engine, and nothing else.
``cactus-needle``'s Python package is a ``ctypes`` wrapper over ``libneedle3``:
``needle_load`` maps the archive, ``needle_init`` tokenizes the system prompt
and the tool schemas into a static prefix, ``needle_complete`` runs one turn and
``needle_reset`` drops the turn state. That is the same C API the Android
``:router`` service drives over JNI, so a host measurement is a measurement of
the artifact rather than of a different representation of the weights.

Passing ``weights=`` (rather than letting the package fetch its own base copy)
is deliberate: the file the harness hashes is then exactly the file the engine
loads. There is no path here that scores a ``.safetensors`` checkpoint or an
adapter.

THE SYSTEM PROMPT CARRIES NO DATE

``cactus-needle`` prepends a ``date: YYYY-MM-DD`` fact to the system prompt by
default. That is switched off, for the reason the spike gives and Vesta repeats
everywhere: **Vesta owns time.** A date in the static prefix is also a prefix
that changes every day, which makes two runs of the same corpus incomparable.
The base model does invent a time for "Wake me up." either way — with the date
fact it invented the current clock time — and that is a finding to record, not
a variable to leave floating.

TELEMETRY IS OFF

The package sends anonymous usage counts by default. Both documented opt-outs
are set before the import, because an offline-first project does not make an
outbound request to benchmark a local file.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from ..evaluate import EngineResponse
from ..manifest import sha256_file
from . import response_from_envelope

__all__ = ["NEEDLE_SYSTEM_PROMPT", "NeedleEngine", "NeedleUnavailable"]

#: The system prompt, byte-identical to the Android spike's
#: (``apps/mobile/lib/router-spike/tools.ts``).
#:
#: Short on purpose. ``needle_init`` tokenizes the system prompt plus every tool
#: schema into a static prefix, so prompt text is context the model pays for on
#: every turn — and a long one would make the prefix-token count say more about
#: our prose than about the model.
NEEDLE_SYSTEM_PROMPT = (
    "You control a phone. Route the user's request to a tool when one fits, "
    "and answer as chat when none does."
)


class NeedleUnavailable(RuntimeError):
    """The runtime could not be imported or the artifact could not be loaded.

    Raised rather than worked around. If ``.cact`` evaluation is not possible
    on this host, the honest outcome is no report at all — a report produced
    some other way would be a number about something else.
    """


def _import_needle(needle_path: str | None):
    # Set before the import: the package reads both at call time, but the
    # first-run notice fires on import.
    os.environ.setdefault("NEEDLE_TELEMETRY", "0")
    os.environ.setdefault("DO_NOT_TRACK", "1")
    if needle_path:
        import sys

        resolved = str(Path(needle_path).resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
    try:
        import needle  # noqa: PLC0415 - deliberately lazy; the core package has no dependencies
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise NeedleUnavailable(
            "cactus-needle is not importable. Install it (`pip install cactus-needle`) or "
            "point --needle-path at a local checkout. The core vesta_router package stays "
            "dependency-free, so this is never imported until an evaluation asks for it."
        ) from exc
    return needle


class NeedleEngine:
    """One loaded ``.cact``, driven one prompt at a time."""

    def __init__(
        self,
        artifact: str | Path,
        tools: list[dict],
        system_prompt: str = NEEDLE_SYSTEM_PROMPT,
        needle_path: str | None = None,
        auto_date: bool = False,
    ) -> None:
        self._artifact = Path(artifact)
        if not self._artifact.is_file():
            raise NeedleUnavailable(f"{self._artifact} does not exist")

        needle = _import_needle(needle_path)
        self._needle_version = getattr(needle, "__version__", None)
        self._system_prompt = system_prompt
        self._auto_date = auto_date
        self._tools_json = json.dumps(tools, ensure_ascii=False)
        self._sha256 = sha256_file(self._artifact)
        self._size_bytes = self._artifact.stat().st_size

        try:
            self._library = needle._library_path(3)
        except Exception:  # pragma: no cover - only the label suffers
            self._library = None

        try:
            self._agent = needle.Needle(
                tools=self._tools_json,
                system=system_prompt,
                weights=str(self._artifact),
                auto_date=auto_date,
            )
        except Exception as exc:
            raise NeedleUnavailable(f"could not load {self._artifact}: {exc}") from exc

    def describe(self) -> dict:
        return {
            "engine": "cactus-needle",
            "engineVersion": self._needle_version,
            "engineLibrary": self._library,
            "artifactPath": str(self._artifact),
            "artifactSha256": self._sha256,
            "artifactSizeBytes": self._size_bytes,
            "artifactKind": "cact",
            "systemPrompt": self._system_prompt,
            "autoDate": self._auto_date,
        }

    @property
    def sha256(self) -> str:
        return self._sha256

    def reset(self) -> bool:
        """Drop turn state. ``False`` means the case that follows is suspect."""
        try:
            self._agent.reset()
            return True
        except Exception:
            return False

    def complete(self, prompt: str) -> EngineResponse:
        started = time.perf_counter()
        try:
            envelope = self._agent.complete(prompt)
        except Exception as exc:
            elapsed = (time.perf_counter() - started) * 1000
            return EngineResponse(latency_ms=round(elapsed, 3), error=f"{type(exc).__name__}: {exc}")
        elapsed = (time.perf_counter() - started) * 1000
        if not isinstance(envelope, dict):
            return EngineResponse(
                latency_ms=round(elapsed, 3),
                error=f"engine returned {type(envelope).__name__}, not an envelope",
            )
        return response_from_envelope(envelope, round(elapsed, 3))

    def close(self) -> None:
        try:
            self._agent.close()
        except Exception:
            pass
