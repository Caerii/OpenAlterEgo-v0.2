"""Discrete motor-unit layer: per-unit channel routing, gains, and label affinity."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def init_motor_unit_layer(
    rng: np.random.Generator,
    n_motor_units: int,
    n_channels: int,
    n_labels: int,
    preset_channel_weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create random MU assignments for synthetic training streams.

    Returns
    -------
    mu_label_id :
        Shape ``(n_motor_units,)``, int in ``0 .. n_labels-1`` — crude "synergy" tag per unit.
    mu_channel_weights :
        Shape ``(n_motor_units, n_channels)``, rows are non-negative and sum to 1 (routing to electrodes).
    mu_gain :
        Shape ``(n_motor_units,)``, log-normal-ish relative spike amplitude (unitless, ~1 mean).
    """
    if n_motor_units < 1:
        raise ValueError("n_motor_units must be >= 1")
    if n_labels < 1:
        raise ValueError("n_labels must be >= 1")

    mu_label_id = rng.integers(0, n_labels, size=n_motor_units, dtype=np.int32)
    if preset_channel_weights is not None:
        w = np.asarray(preset_channel_weights, dtype=np.float64)
        if w.shape != (n_motor_units, n_channels):
            raise ValueError(f"preset_channel_weights shape {w.shape} != {(n_motor_units, n_channels)}")
        w = np.maximum(w, 0.0)
        w /= np.sum(w, axis=1, keepdims=True) + 1e-12
    else:
        w = rng.random((n_motor_units, n_channels), dtype=np.float64)
        w /= np.sum(w, axis=1, keepdims=True) + 1e-12
    gain = rng.lognormal(mean=0.0, sigma=0.22, size=n_motor_units).astype(np.float64)
    gain = gain / (np.mean(gain) + 1e-12)
    return mu_label_id, w.astype(np.float32), gain.astype(np.float32)


def firing_rates_token_segment(
    mu_label_id: np.ndarray,
    mu_gain: np.ndarray,
    *,
    active_label_id: int,
    token_firing_rate_hz: float,
    baseline_firing_rate_hz: float,
    off_label_rate_scale: float,
) -> np.ndarray:
    """Per-MU Poisson rates (Hz) during a token-aligned segment.

    Units whose label matches ``active_label_id`` share ``token_firing_rate_hz`` in proportion to
    ``mu_gain``. Other units fire at a suppressed baseline level (``off_label_rate_scale``).
    """
    n = mu_label_id.size
    g = np.asarray(mu_gain, dtype=np.float64)
    mask = mu_label_id == int(active_label_id)
    rates = np.zeros(n, dtype=np.float64)
    s_act = float(np.sum(g[mask]))
    if s_act > 1e-12:
        rates[mask] = float(token_firing_rate_hz) * g[mask] / s_act
    # Suppressed units: share scaled baseline
    n_off = int(np.sum(~mask))
    if n_off > 0:
        s_off = float(np.sum(g[~mask])) + 1e-12
        rates[~mask] = float(baseline_firing_rate_hz) * float(off_label_rate_scale) * g[~mask] / s_off
    return rates


def firing_rates_baseline_segment(mu_gain: np.ndarray, baseline_firing_rate_hz: float) -> np.ndarray:
    """Per-MU rates (Hz) during rest; total rate ≈ ``baseline_firing_rate_hz``."""
    g = np.asarray(mu_gain, dtype=np.float64)
    s = float(np.sum(g)) + 1e-12
    return float(baseline_firing_rate_hz) * g / s
