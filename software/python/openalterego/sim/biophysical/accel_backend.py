"""Runtime selection among python / numba / rust motor-pool backends."""

from __future__ import annotations

from typing import Literal, Optional

AccelBackend = Literal["auto", "python", "numba", "rust"]

_RESOLVED: Optional[str] = None
_RUST_AVAILABLE: Optional[bool] = None
_NUMBA_AVAILABLE: Optional[bool] = None


def _probe_rust() -> bool:
    global _RUST_AVAILABLE
    if _RUST_AVAILABLE is None:
        try:
            import openalterego_accel  # noqa: F401

            _RUST_AVAILABLE = True
        except ImportError:
            _RUST_AVAILABLE = False
    return bool(_RUST_AVAILABLE)


def _probe_numba() -> bool:
    global _NUMBA_AVAILABLE
    if _NUMBA_AVAILABLE is None:
        try:
            from . import pool_numba

            _NUMBA_AVAILABLE = bool(pool_numba.HAS_NUMBA)
        except ImportError:
            _NUMBA_AVAILABLE = False
    return bool(_NUMBA_AVAILABLE)


def resolve_backend(requested: str) -> str:
    """Return concrete backend: ``python``, ``numba``, or ``rust``."""
    req = str(requested or "auto").strip().lower()
    if req in ("fast", "auto"):
        if _probe_numba():
            return "numba"
        if _probe_rust():
            return "rust"
        return "python"
    if req == "rust":
        if not _probe_rust():
            raise RuntimeError(
                "synth_mode=rust requires openalterego_accel; "
                "build with: cd software/python/accel && maturin develop --release"
            )
        return "rust"
    if req == "numba":
        if not _probe_numba():
            raise RuntimeError("synth_mode=numba requires numba; install with: uv add numba")
        return "numba"
    if req in ("python", "batch", "legacy"):
        return "python"
    raise ValueError(f"unknown synth_mode/accel backend: {requested!r}")


def active_backend_label() -> str:
    """Best backend available without raising."""
    return resolve_backend("auto")
