"""Travel-time delays across the electrode array (conduction velocity proxy)."""

from __future__ import annotations

import numpy as np


def channel_delays_from_pickup(
    channel_weights: np.ndarray,
    fs_hz: float,
    rng: np.random.Generator,
    *,
    velocity_m_s: float = 4.0,
    array_span_m: float = 0.042,
    jitter_samples: int = 1,
    max_delay_ms: float = 12.0,
) -> np.ndarray:
    """Integer per-channel delay (samples) for one motor unit, shape (n_channels,)."""
    w = np.asarray(channel_weights, dtype=np.float64).ravel()
    c = int(w.size)
    if c <= 0:
        return np.zeros((0,), dtype=np.int32)
    anchor = int(np.argmax(w))
    dx = float(array_span_m) / max(c - 1, 1)
    dist_m = np.abs(np.arange(c, dtype=np.float64) - float(anchor)) * dx
    v = max(float(velocity_m_s), 0.4)
    base = (dist_m / v * float(fs_hz)).astype(np.float64)
    jm = max(0, int(jitter_samples))
    if jm > 0:
        base = base + rng.integers(-jm, jm + 1, size=c).astype(np.float64)
    max_s = max(0, int(float(fs_hz) * float(max_delay_ms) / 1000.0))
    d = np.clip(np.round(base), 0, max_s).astype(np.int32)
    return d
