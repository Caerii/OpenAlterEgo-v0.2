"""Simplified MUAP-like waveforms (time domain).

Real surface MUAPs depend on electrode geometry, fiber depth, and conduction velocity; here we use
a minimal skewed biphasic shape (rise + decay) with duration on the order of ~5-15 ms,
which matches the coarse scale often cited for sEMG MUAP width (Farina and Merletti style refs).
"""

from __future__ import annotations

import numpy as np


def bipolar_muap_template(
    fs_hz: float,
    *,
    duration_ms: float = 10.0,
    rise_ms: float = 2.5,
    decay_ms: float = 7.5,
    asymmetry: float = 0.35,
) -> np.ndarray:
    """Return a normalized MUAP-like waveform (unit peak magnitude), length ~ duration_ms."""
    dur_s = max(duration_ms, rise_ms + decay_ms) / 1000.0
    n = max(8, int(fs_hz * dur_s))
    t = np.arange(n, dtype=np.float64) / float(fs_hz)
    tr = max(rise_ms / 1000.0, 1e-6)
    td = max(decay_ms / 1000.0, 1e-6)
    main = (1.0 - np.exp(-t / tr)) * np.exp(-t / td)
    main = main.astype(np.float64)
    peak = float(np.max(np.abs(main))) + 1e-12
    main /= peak
    lag = max(1, int(0.15 * n))
    secondary = np.zeros_like(main)
    if lag < n:
        secondary[lag:] = -asymmetry * main[:-lag]
    y = main + secondary
    y = (y / (np.max(np.abs(y)) + 1e-12)).astype(np.float32)
    return y


def stretch_muap_template(base: np.ndarray, *, width_scale: float) -> np.ndarray:
    """Time-stretch a normalized MUAP template (``width_scale`` > 1 → wider waveform).

    Uses linear interpolation on a fixed relative time grid; re-normalizes peak magnitude.
    """
    b = np.asarray(base, dtype=np.float64).ravel()
    if b.size < 4:
        return np.asarray(base, dtype=np.float32)
    ws = float(np.clip(width_scale, 0.55, 2.2))
    n0 = int(b.size)
    n1 = max(8, int(round(n0 * ws)))
    t0 = np.linspace(0.0, 1.0, n0, dtype=np.float64)
    t1 = np.linspace(0.0, 1.0, n1, dtype=np.float64)
    y = np.interp(t1, t0, b).astype(np.float64)
    peak = float(np.max(np.abs(y))) + 1e-12
    y = (y / peak).astype(np.float32)
    return y
