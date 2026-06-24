"""Flat spike-event buffers + batched scatter for one-call-per-chunk synthesis."""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np

from .pool_fast import MotorPoolSynthCache, _spike_train_indices


def collect_spike_events(
    rng: np.random.Generator,
    rates_hz: np.ndarray,
    amplitudes_uV: np.ndarray,
    cache: MotorPoolSynthCache,
    *,
    n: int,
    fs_hz: float,
    envelope: Optional[np.ndarray],
    time_jitter_std_s: float,
    refractory_samples: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return ``(mu_idx, spike_t0, spike_amp)`` for all units; amps include envelope."""
    n_mu = int(rates_hz.size)
    rates = np.asarray(rates_hz, dtype=np.float64).ravel()
    amps = np.asarray(amplitudes_uV, dtype=np.float64).ravel()
    env = np.ones((n,), dtype=np.float32) if envelope is None else np.asarray(envelope, dtype=np.float32).ravel()
    refr = (
        np.zeros(n_mu, dtype=np.int32)
        if refractory_samples is None
        else np.asarray(refractory_samples, dtype=np.int32).ravel()
    )
    jit = float(time_jitter_std_s)

    mu_parts: list[np.ndarray] = []
    t0_parts: list[np.ndarray] = []
    amp_parts: list[np.ndarray] = []
    total = 0

    for i in range(n_mu):
        idx = _spike_train_indices(
            rng,
            float(rates[i]),
            n,
            float(fs_hz),
            refractory_samples=int(refr[i]),
            time_jitter_std_s=jit,
        )
        if idx.size == 0:
            continue
        total += int(idx.size)
        k = int(idx.size)
        mu_parts.append(np.full((k,), i, dtype=np.int32))
        t0_parts.append(idx.astype(np.int32, copy=False))
        amp_parts.append((float(amps[i]) * env[idx.astype(np.intp)]).astype(np.float32))

    if total == 0:
        empty_i = np.zeros((0,), dtype=np.int32)
        empty_f = np.zeros((0,), dtype=np.float32)
        return empty_i, empty_i, empty_f, 0

    return (
        np.concatenate(mu_parts),
        np.concatenate(t0_parts),
        np.concatenate(amp_parts),
        total,
    )
