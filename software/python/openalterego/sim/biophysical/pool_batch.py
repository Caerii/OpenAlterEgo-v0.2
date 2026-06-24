"""Vectorized motor-pool MUAP superposition (batch spike placement)."""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np


def _apply_refractory(indices: np.ndarray, refractory_samples: int) -> np.ndarray:
    """Drop spikes closer than ``refractory_samples`` (sorted input)."""
    if refractory_samples <= 0 or indices.size <= 1:
        return indices
    keep = [int(indices[0])]
    for idx in indices[1:]:
        if int(idx) - keep[-1] >= refractory_samples:
            keep.append(int(idx))
    return np.asarray(keep, dtype=np.int32)


def _poisson_spike_indices(
    rng: np.random.Generator,
    rate_hz: float,
    n_samples: int,
    fs_hz: float,
    *,
    refractory_samples: int = 0,
    time_jitter_std_s: float = 0.0,
) -> np.ndarray:
    if rate_hz <= 0.0 or n_samples <= 0:
        return np.zeros((0,), dtype=np.int32)
    T = float(n_samples) / float(fs_hz)
    k = int(rng.poisson(max(0.0, float(rate_hz)) * T))
    if k <= 0:
        return np.zeros((0,), dtype=np.int32)
    times = rng.uniform(0.0, T, size=k).astype(np.float64)
    if time_jitter_std_s > 0.0:
        times += rng.normal(0.0, float(time_jitter_std_s), size=k)
    times = np.clip(times, 0.0, max(T * (1.0 - 1e-9), 0.0))
    idx = (times * float(fs_hz)).astype(np.int32)
    idx = np.clip(idx, 0, n_samples - 1)
    idx.sort()
    return _apply_refractory(idx, int(refractory_samples))


def superpose_motor_pool_window(
    x: np.ndarray,
    fs_hz: float,
    rng: np.random.Generator,
    rates_hz: np.ndarray,
    amplitudes_uV: np.ndarray,
    channel_weights: np.ndarray,
    muap_templates: Sequence[np.ndarray],
    *,
    envelope: Optional[np.ndarray] = None,
    time_jitter_std_s: float = 0.0,
    refractory_samples: Optional[np.ndarray] = None,
    spread_across_channels: bool = True,
    channel_delays: Optional[np.ndarray] = None,
) -> int:
    """Synthesize all motor units into ``x`` (n_samples, n_channels); returns spike count."""
    n, c = x.shape
    n_mu = int(rates_hz.size)
    if n_mu == 0 or n <= 0:
        return 0

    rates = np.asarray(rates_hz, dtype=np.float64).ravel()
    amps = np.asarray(amplitudes_uV, dtype=np.float64).ravel()
    w = np.asarray(channel_weights, dtype=np.float32)
    if w.shape != (n_mu, c):
        raise ValueError(f"channel_weights shape {w.shape} != ({n_mu}, {c})")

    env = np.ones((n,), dtype=np.float32) if envelope is None else np.asarray(envelope, dtype=np.float32).ravel()
    if env.size != n:
        raise ValueError(f"envelope length {env.size} != {n}")

    refr = None
    if refractory_samples is not None:
        refr = np.asarray(refractory_samples, dtype=np.int32).ravel()
        if refr.size != n_mu:
            raise ValueError(f"refractory_samples length {refr.size} != {n_mu}")

    dly = None
    if channel_delays is not None:
        dly = np.asarray(channel_delays, dtype=np.int32)
        if dly.shape != (n_mu, c):
            raise ValueError(f"channel_delays shape {dly.shape} != ({n_mu}, {c})")

    jit = float(time_jitter_std_s)
    total_spikes = 0

    for i in range(n_mu):
        ref_i = int(refr[i]) if refr is not None else 0
        idx = _poisson_spike_indices(
            rng, float(rates[i]), n, float(fs_hz), refractory_samples=ref_i, time_jitter_std_s=jit
        )
        if idx.size == 0:
            continue
        total_spikes += int(idx.size)
        m = np.asarray(muap_templates[i], dtype=np.float32)
        L = int(m.shape[0])
        wi = w[i]
        amp_base = float(amps[i])
        if spread_across_channels:
            if dly is not None:
                di = dly[i]
                for idx0 in idx:
                    a = amp_base * float(env[int(idx0)])
                    for ch in range(c):
                        start = int(idx0) + int(di[ch])
                        if start < 0 or start >= n:
                            continue
                        end = min(n, start + L)
                        sl = end - start
                        if sl <= 0:
                            continue
                        x[start:end, ch] += (a * float(wi[ch]) * m[:sl]).astype(np.float32, copy=False)
            else:
                for idx0 in idx:
                    a = amp_base * float(env[int(idx0)])
                    end = min(n, int(idx0) + L)
                    sl = end - int(idx0)
                    if sl <= 0:
                        continue
                    x[int(idx0) : end, :] += (a * m[:sl, None] * wi[None, :]).astype(np.float32, copy=False)
        else:
            probs = wi.astype(np.float64)
            probs /= probs.sum() + 1e-12
            ch_pick = rng.choice(c, size=idx.size, p=probs)
            for k, idx0 in enumerate(idx):
                a = amp_base * float(env[int(idx0)])
                end = min(n, int(idx0) + L)
                sl = end - int(idx0)
                if sl <= 0:
                    continue
                x[int(idx0) : end, int(ch_pick[k])] += (a * m[:sl]).astype(np.float32, copy=False)

    return total_spikes


def stack_templates(templates: List[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Pad templates to ``(n_mu, L_max)`` and return lengths ``(n_mu,)``."""
    if not templates:
        return np.zeros((0, 1), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    lengths = np.array([int(t.shape[0]) for t in templates], dtype=np.int32)
    L_max = int(lengths.max())
    n_mu = len(templates)
    out = np.zeros((n_mu, L_max), dtype=np.float32)
    for i, t in enumerate(templates):
        out[i, : int(t.shape[0])] = np.asarray(t, dtype=np.float32)
    return out, lengths
