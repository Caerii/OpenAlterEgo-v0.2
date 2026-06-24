"""Lightweight augmentation on σ(τ) sequences during training."""

from __future__ import annotations

import numpy as np


def augment_sigma_sequence(
    seq: np.ndarray,
    rng: np.random.Generator,
    *,
    noise_std: float = 0.02,
    time_mask_frac: float = 0.08,
    feat_mask_frac: float = 0.05,
) -> np.ndarray:
    """Time/feature masking + Gaussian noise (SpecAugment-style)."""
    x = np.asarray(seq, dtype=np.float32).copy()
    t, d = x.shape
    if t <= 0 or d <= 0:
        return x

    if float(noise_std) > 0:
        x += rng.normal(0.0, float(noise_std), size=x.shape).astype(np.float32)

    n_mask_t = max(1, int(round(t * float(time_mask_frac)))) if t > 4 else 0
    for _ in range(n_mask_t):
        w = int(rng.integers(1, max(2, t // 8 + 1)))
        s0 = int(rng.integers(0, max(1, t - w)))
        x[s0 : s0 + w, :] = 0.0

    n_mask_f = max(1, int(round(d * float(feat_mask_frac)))) if d > 4 else 0
    for _ in range(n_mask_f):
        j = int(rng.integers(0, d))
        x[:, j] = 0.0

    return x
