"""Simplified recruitment: small (low-gain) units preferentially active at low effort.

This is a **toy** Henneman ordering: we rank units by ``mu_gain`` ascending (proxy for low
threshold / early recruitment) and scale rates with a smooth gate from scalar ``activation``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def recruitment_rate_multipliers(
    mu_gain: np.ndarray,
    activation: float,
    *,
    steepness: float = 14.0,
    recruitment_rank: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return per-MU multipliers in ``(0, 1]``, shape ``(n_motor_units,)``.

    ``activation`` in ``[0, 1]``: low → mostly small units; high → full pool participates.
    When ``recruitment_rank`` is set (physiological pool), ranks drive Henneman ordering directly.
    """
    g = np.asarray(mu_gain, dtype=np.float64).ravel()
    n = int(g.size)
    if n == 0:
        return np.zeros((0,), dtype=np.float64)
    if recruitment_rank is not None:
        ranks = np.asarray(recruitment_rank, dtype=np.float64).ravel()
        if ranks.size != n:
            raise ValueError(f"recruitment_rank length {ranks.size} != {n}")
    else:
        order = np.argsort(g)
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.linspace(0.0, 1.0, n, dtype=np.float64)
    a = float(np.clip(activation, 0.0, 1.0))
    k = max(float(steepness), 1.0)
    z = k * (a - ranks)
    z = np.clip(z, -60.0, 60.0)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float64)
