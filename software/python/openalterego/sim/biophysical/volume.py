"""Electrode pickup / volume-style mixing (linear, row-stochastic).

Full volume conduction is PDE-scale; this module provides a **fixed positive mixing matrix**
(blurred spatial pickup + mild per-electrode gain jitter) so multi-channel sEMG is not limited
to independent channels with nearest-neighbor crosstalk only.
"""

from __future__ import annotations

import numpy as np


def diffuse_electrode_mix(
    n_channels: int,
    rng: np.random.Generator,
    *,
    neighbor_weight: float = 0.14,
    diffuse_length: float = 2.4,
    diffuse_strength: float = 0.55,
    channel_gain_log_sigma: float = 0.06,
) -> np.ndarray:
    """Return ``(c, c)`` row-stochastic mixing matrix (float32).

    Combines tri-diagonal neighbor leakage with an exponential falloff in electrode index
    (crude stand-in for spatial smearing) and log-normal per-channel gains.
    """
    c = int(n_channels)
    if c <= 0:
        raise ValueError("n_channels must be positive")
    nw = max(0.0, float(neighbor_weight))
    M = np.eye(c, dtype=np.float64)
    for i in range(c):
        if i - 1 >= 0:
            M[i, i - 1] += nw
        if i + 1 < c:
            M[i, i + 1] += nw
    pos = np.arange(c, dtype=np.float64)
    d = np.abs(pos[:, None] - pos[None, :])
    ell = max(float(diffuse_length), 0.25)
    blur = float(diffuse_strength) * np.exp(-d / ell)
    M = M + blur
    g = np.exp(rng.normal(0.0, float(channel_gain_log_sigma), size=(c,)))
    M = (M * g[None, :]).astype(np.float64)
    rs = M.sum(axis=1, keepdims=True) + 1e-12
    M = M / rs
    return M.astype(np.float32)
