"""Discrete 1D muscle to electrode forward pickup."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def green_pickup_matrix(
    n_electrodes: int,
    n_sources: int,
    *,
    falloff: float = 0.16,
) -> np.ndarray:
    """Return G (n_electrodes, n_sources), columns nonnegative, column-sum 1."""
    ne = int(n_electrodes)
    ns = int(n_sources)
    if ne < 1 or ns < 1:
        raise ValueError("n_electrodes and n_sources must be >= 1")
    ell = max(float(falloff), 0.03)
    xe = np.linspace(0.0, 1.0, ne, dtype=np.float64)
    xs = np.linspace(0.0, 1.0, ns, dtype=np.float64)
    d = np.abs(xe[:, None] - xs[None, :])
    g = np.exp(-d / ell)
    g /= np.sum(g, axis=0, keepdims=True) + 1e-12
    return g.astype(np.float32)


def motor_unit_pickup_weights(
    rng: np.random.Generator,
    G: np.ndarray,
    n_motor_units: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (mu_source_idx, mu_channel_weights) with rows summing to 1."""
    _, ns = G.shape
    idx = rng.integers(0, ns, size=int(n_motor_units), dtype=np.int32)
    w = np.stack([G[:, int(j)].astype(np.float64) for j in idx], axis=0)
    w = np.maximum(w, 0.0)
    w /= np.sum(w, axis=1, keepdims=True) + 1e-12
    return idx, w.astype(np.float32)
