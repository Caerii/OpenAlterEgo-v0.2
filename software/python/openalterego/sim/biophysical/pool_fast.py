"""High-throughput motor-pool synthesis (convolution + batched spike scatter).

Scaling (dominant terms, no conduction delays):
  T_motor ~ O(n_mu * n * log n) via FFT convolve, or O(n_spikes * L * c) via scatter
  n_spikes ~ sum_i rate_i * (n / fs)

Use ``MotorPoolSynthCache`` to amortize template padding across chunks.
Backends: python (default), numba (``pool_numba``), rust (``openalterego_accel``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
from scipy import signal

from .accel_backend import resolve_backend
from .pool_numba import min_convolve_samples


def _refractory_mask(indices: np.ndarray, refractory_samples: int) -> np.ndarray:
    if refractory_samples <= 0 or indices.size <= 1:
        return indices
    out: List[int] = [int(indices[0])]
    last = out[0]
    ref = int(refractory_samples)
    for v in indices[1:]:
        vi = int(v)
        if vi - last >= ref:
            out.append(vi)
            last = vi
    return np.asarray(out, dtype=np.int32)


def _spike_train_indices(
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
    times = rng.uniform(0.0, T, size=k)
    if time_jitter_std_s > 0.0:
        times += rng.normal(0.0, float(time_jitter_std_s), size=k)
    times = np.clip(times, 0.0, max(T * (1.0 - 1e-9), 0.0))
    idx = np.floor(times * float(fs_hz)).astype(np.int32)
    idx = np.clip(idx, 0, n_samples - 1)
    idx.sort()
    return _refractory_mask(idx, refractory_samples)


@dataclass
class MotorPoolSynthCache:
    """Pre-padded templates and weights for repeated chunk synthesis."""

    templates: np.ndarray  # (n_mu, L_max) float32
    lengths: np.ndarray  # (n_mu,) int32
    weights: np.ndarray  # (n_mu, c) float32
    channel_delays: Optional[np.ndarray]  # (n_mu, c) int32 or None

    @classmethod
    def from_pool(
        cls,
        muap_templates: Sequence[np.ndarray],
        channel_weights: np.ndarray,
        channel_delays: Optional[np.ndarray] = None,
    ) -> MotorPoolSynthCache:
        w = np.asarray(channel_weights, dtype=np.float32)
        lengths = np.array([int(t.shape[0]) for t in muap_templates], dtype=np.int32)
        L_max = max(1, int(lengths.max()) if lengths.size else 1)
        n_mu = len(muap_templates)
        tpl = np.zeros((n_mu, L_max), dtype=np.float32)
        for i, t in enumerate(muap_templates):
            ti = np.asarray(t, dtype=np.float32).ravel()
            tpl[i, : ti.size] = ti
        dly = None
        if channel_delays is not None:
            dly = np.asarray(channel_delays, dtype=np.int32)
        return cls(templates=tpl, lengths=lengths, weights=w, channel_delays=dly)


def superpose_motor_pool_convolve(
    x: np.ndarray,
    fs_hz: float,
    rng: np.random.Generator,
    rates_hz: np.ndarray,
    amplitudes_uV: np.ndarray,
    cache: MotorPoolSynthCache,
    *,
    envelope: Optional[np.ndarray] = None,
    time_jitter_std_s: float = 0.0,
    refractory_samples: Optional[np.ndarray] = None,
) -> int:
    """FFT-friendly path: no per-channel conduction delays (``cache.channel_delays is None``)."""
    n, c = x.shape
    n_mu = int(rates_hz.size)
    if n_mu == 0 or cache.channel_delays is not None:
        return 0

    rates = np.asarray(rates_hz, dtype=np.float64).ravel()
    amps = np.asarray(amplitudes_uV, dtype=np.float64).ravel()
    env = np.ones((n,), dtype=np.float32) if envelope is None else np.asarray(envelope, dtype=np.float32).ravel()
    refr = (
        np.zeros(n_mu, dtype=np.int32)
        if refractory_samples is None
        else np.asarray(refractory_samples, dtype=np.int32).ravel()
    )
    jit = float(time_jitter_std_s)
    total = 0
    tpl = cache.templates
    w = cache.weights

    for i in range(n_mu):
        idx = _spike_train_indices(
            rng, float(rates[i]), n, float(fs_hz),
            refractory_samples=int(refr[i]), time_jitter_std_s=jit,
        )
        if idx.size == 0:
            continue
        total += int(idx.size)
        train = np.zeros((n,), dtype=np.float32)
        train[idx] = (amps[i] * env[idx]).astype(np.float32)
        Li = int(cache.lengths[i])
        m = tpl[i, :Li]
        if Li < 4:
            continue
        y = signal.fftconvolve(train, m, mode="same").astype(np.float32, copy=False)
        x += y[:, None] * w[i, None, :]
    return total


def superpose_motor_pool_scatter(
    x: np.ndarray,
    fs_hz: float,
    rng: np.random.Generator,
    rates_hz: np.ndarray,
    amplitudes_uV: np.ndarray,
    cache: MotorPoolSynthCache,
    *,
    envelope: Optional[np.ndarray] = None,
    time_jitter_std_s: float = 0.0,
    refractory_samples: Optional[np.ndarray] = None,
    spread_across_channels: bool = True,
) -> int:
    """Batched spike scatter; supports per-channel delays via ``cache.channel_delays``."""
    n, c = x.shape
    n_mu = int(rates_hz.size)
    if n_mu == 0:
        return 0

    rates = np.asarray(rates_hz, dtype=np.float64).ravel()
    amps = np.asarray(amplitudes_uV, dtype=np.float64).ravel()
    env = np.ones((n,), dtype=np.float32) if envelope is None else np.asarray(envelope, dtype=np.float32).ravel()
    refr = (
        np.zeros(n_mu, dtype=np.int32)
        if refractory_samples is None
        else np.asarray(refractory_samples, dtype=np.int32).ravel()
    )
    jit = float(time_jitter_std_s)
    tpl = cache.templates
    w = cache.weights
    dly = cache.channel_delays
    total = 0

    for i in range(n_mu):
        idx = _spike_train_indices(
            rng, float(rates[i]), n, float(fs_hz),
            refractory_samples=int(refr[i]), time_jitter_std_s=jit,
        )
        if idx.size == 0:
            continue
        total += int(idx.size)
        Li = int(cache.lengths[i])
        m = tpl[i, :Li]
        wi = w[i]
        amp_base = float(amps[i])
        if spread_across_channels and dly is not None:
            di = dly[i]
            for t0 in idx:
                a = amp_base * float(env[int(t0)])
                for ch in range(c):
                    start = int(t0) + int(di[ch])
                    if start < 0 or start >= n:
                        continue
                    end = min(n, start + Li)
                    sl = end - start
                    if sl <= 0:
                        continue
                    x[start:end, ch] += (a * float(wi[ch]) * m[:sl]).astype(np.float32, copy=False)
        elif spread_across_channels:
            for t0 in idx:
                a = amp_base * float(env[int(t0)])
                end = min(n, int(t0) + Li)
                sl = end - int(t0)
                if sl <= 0:
                    continue
                x[int(t0):end, :] += (a * m[:sl, None] * wi[None, :]).astype(np.float32, copy=False)
        else:
            probs = wi.astype(np.float64)
            probs /= probs.sum() + 1e-12
            picks = rng.choice(c, size=idx.size, p=probs)
            for k, t0 in enumerate(idx):
                a = amp_base * float(env[int(t0)])
                end = min(n, int(t0) + Li)
                sl = end - int(t0)
                if sl <= 0:
                    continue
                x[int(t0):end, int(picks[k])] += (a * m[:sl]).astype(np.float32, copy=False)
    return total


def _superpose_rust(
    x: np.ndarray,
    fs_hz: float,
    rng: np.random.Generator,
    rates_hz: np.ndarray,
    amplitudes_uV: np.ndarray,
    cache: MotorPoolSynthCache,
    *,
    envelope: Optional[np.ndarray] = None,
    time_jitter_std_s: float = 0.0,
    refractory_samples: Optional[np.ndarray] = None,
    spread_across_channels: bool = True,
) -> int:
    import openalterego_accel

    from .pool_events import collect_spike_events

    n, c = x.shape
    if int(rates_hz.size) == 0:
        return 0

    if not spread_across_channels:
        return superpose_motor_pool_scatter(
            x, fs_hz, rng, rates_hz, amplitudes_uV, cache,
            envelope=envelope, time_jitter_std_s=time_jitter_std_s,
            refractory_samples=refractory_samples, spread_across_channels=False,
        )

    mu_idx, spike_t0, spike_amp, total = collect_spike_events(
        rng,
        rates_hz,
        amplitudes_uV,
        cache,
        n=n,
        fs_hz=float(fs_hz),
        envelope=envelope,
        time_jitter_std_s=time_jitter_std_s,
        refractory_samples=refractory_samples,
    )
    if total == 0:
        return 0

    dly = cache.channel_delays
    if dly is None:
        dly_pad = np.zeros((cache.templates.shape[0], c), dtype=np.int32)
        has_delays = False
    else:
        dly_pad = np.asarray(dly, dtype=np.int32)
        has_delays = True

    if not hasattr(openalterego_accel, "scatter_pool_batched"):
        from .pool_numba import superpose_motor_pool_numba

        return superpose_motor_pool_numba(
            x, fs_hz, rng, rates_hz, amplitudes_uV, cache,
            envelope=envelope, time_jitter_std_s=time_jitter_std_s,
            refractory_samples=refractory_samples,
            spread_across_channels=spread_across_channels,
        )

    openalterego_accel.scatter_pool_batched(
        x,
        mu_idx,
        spike_t0,
        spike_amp,
        cache.templates,
        cache.lengths,
        cache.weights,
        dly_pad,
        n,
        c,
        has_delays,
    )
    return total


def superpose_motor_pool_fast(
    x: np.ndarray,
    fs_hz: float,
    rng: np.random.Generator,
    rates_hz: np.ndarray,
    amplitudes_uV: np.ndarray,
    cache: MotorPoolSynthCache,
    *,
    envelope: Optional[np.ndarray] = None,
    time_jitter_std_s: float = 0.0,
    refractory_samples: Optional[np.ndarray] = None,
    spread_across_channels: bool = True,
    prefer_convolve: bool = True,
    backend: str = "auto",
) -> int:
    """Dispatch to rust/numba scatter, python convolve, or python scatter."""
    resolved = resolve_backend(backend)
    n, _c = x.shape
    min_conv = min_convolve_samples(float(fs_hz))

    if resolved == "rust":
        return _superpose_rust(
            x, fs_hz, rng, rates_hz, amplitudes_uV, cache,
            envelope=envelope, time_jitter_std_s=time_jitter_std_s,
            refractory_samples=refractory_samples,
            spread_across_channels=spread_across_channels,
        )

    if resolved == "numba":
        from .pool_numba import superpose_motor_pool_numba

        return superpose_motor_pool_numba(
            x, fs_hz, rng, rates_hz, amplitudes_uV, cache,
            envelope=envelope, time_jitter_std_s=time_jitter_std_s,
            refractory_samples=refractory_samples,
            spread_across_channels=spread_across_channels,
            spike_indices_fn=_spike_train_indices,
        )

    if (
        prefer_convolve
        and n >= min_conv
        and spread_across_channels
        and cache.channel_delays is None
        and int(cache.templates.shape[1]) >= 4
    ):
        return superpose_motor_pool_convolve(
            x, fs_hz, rng, rates_hz, amplitudes_uV, cache,
            envelope=envelope, time_jitter_std_s=time_jitter_std_s,
            refractory_samples=refractory_samples,
        )
    return superpose_motor_pool_scatter(
        x, fs_hz, rng, rates_hz, amplitudes_uV, cache,
        envelope=envelope, time_jitter_std_s=time_jitter_std_s,
        refractory_samples=refractory_samples,
        spread_across_channels=spread_across_channels,
    )
