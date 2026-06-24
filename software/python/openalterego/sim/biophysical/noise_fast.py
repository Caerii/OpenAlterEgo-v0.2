"""Vectorized sensor-noise paths (AR filters, motion bursts)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import signal

from ..realism import realism_preset_params


def _ar1_chunk(
    innov: np.ndarray,
    phi: float,
    state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter innovations with AR(1); ``innov`` shape ``(n, c)``."""
    n, c = innov.shape
    if n == 0:
        return innov.astype(np.float32, copy=False), state.astype(np.float32, copy=False)
    b = np.array([1.0], dtype=np.float64)
    a = np.array([1.0, -float(phi)], dtype=np.float64)
    out = np.empty((n, c), dtype=np.float32)
    s_out = state.astype(np.float64).copy()
    for ch in range(c):
        y, zf = signal.lfilter(b, a, innov[:, ch].astype(np.float64), zi=[s_out[ch]])
        out[:, ch] = y.astype(np.float32, copy=False)
        s_out[ch] = float(zf[0])
    return out, s_out.astype(np.float32)


def apply_motion_burst_vectorized(
    x: np.ndarray,
    *,
    chunk_start: int,
    fs: float,
    motion_left: int,
    motion_total: int,
    motion_amp: float,
    motion_freq_hz: float,
    motion_phase: float,
    shared_fraction: float,
    line_phases: np.ndarray,
    ch_weights: np.ndarray,
) -> int:
    """Add motion sinusoid for up to ``motion_left`` samples; returns remaining motion samples."""
    n, c = x.shape
    if motion_left <= 0 or motion_amp <= 0.0:
        return motion_left
    steps = min(n, motion_left)
    shared_frac = float(np.clip(shared_fraction, 0.0, 1.0))
    t = (float(chunk_start) + np.arange(steps, dtype=np.float64)) / float(fs)
    env = np.sqrt(np.maximum(motion_left - np.arange(steps), 0) / float(max(motion_total, 1)))
    shared = np.sin(2.0 * math.pi * motion_freq_hz * t + motion_phase)
    local = np.sin(
        2.0 * math.pi * motion_freq_hz * t[:, None] + motion_phase + line_phases[None, :].astype(np.float64)
    )
    mot = motion_amp * env[:, None] * (shared_frac * shared[:, None] + (1.0 - shared_frac) * local)
    x[:steps, :] += (mot * ch_weights[None, :]).astype(np.float32, copy=False)
    return motion_left - steps
