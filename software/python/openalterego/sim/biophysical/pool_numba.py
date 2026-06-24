"""Numba-accelerated spike scatter kernels for motor-pool synthesis."""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    from numba import njit

    HAS_NUMBA = True
except ImportError:  # pragma: no cover - exercised when numba absent
    HAS_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore[misc]
        def _wrap(fn):
            return fn

        return _wrap


if HAS_NUMBA:

    @njit(cache=True, fastmath=True)
    def scatter_unit_multichannel(
        x: np.ndarray,
        spike_idx: np.ndarray,
        amp_base: float,
        env: np.ndarray,
        m: np.ndarray,
        wi: np.ndarray,
        n: int,
        c: int,
        li: int,
    ) -> None:
        for k in range(spike_idx.shape[0]):
            t0 = int(spike_idx[k])
            a = amp_base * float(env[t0])
            end = t0 + li
            if end > n:
                end = n
            sl = end - t0
            if sl <= 0:
                continue
            for ch in range(c):
                w = float(wi[ch])
                for j in range(sl):
                    x[t0 + j, ch] += a * w * float(m[j])

    @njit(cache=True, fastmath=True)
    def scatter_unit_delayed(
        x: np.ndarray,
        spike_idx: np.ndarray,
        amp_base: float,
        env: np.ndarray,
        m: np.ndarray,
        wi: np.ndarray,
        di: np.ndarray,
        n: int,
        c: int,
        li: int,
    ) -> None:
        for k in range(spike_idx.shape[0]):
            t0 = int(spike_idx[k])
            a = amp_base * float(env[t0])
            for ch in range(c):
                start = t0 + int(di[ch])
                if start < 0 or start >= n:
                    continue
                end = start + li
                if end > n:
                    end = n
                sl = end - start
                if sl <= 0:
                    continue
                w = float(wi[ch])
                for j in range(sl):
                    x[start + j, ch] += a * w * float(m[j])

    @njit(cache=True, fastmath=True)
    def scatter_pool_batched(
        x: np.ndarray,
        mu_idx: np.ndarray,
        spike_t0: np.ndarray,
        spike_amp: np.ndarray,
        tpl: np.ndarray,
        lengths: np.ndarray,
        w: np.ndarray,
        dly: np.ndarray,
        n: int,
        c: int,
        has_delays: int,
    ) -> None:
        n_events = mu_idx.shape[0]
        for e in range(n_events):
            i = int(mu_idx[e])
            t0 = int(spike_t0[e])
            if t0 < 0 or t0 >= n:
                continue
            a = float(spike_amp[e])
            li = int(lengths[i])
            if li <= 0:
                continue
            if has_delays != 0:
                for ch in range(c):
                    start = t0 + int(dly[i, ch])
                    if start < 0 or start >= n:
                        continue
                    end = start + li
                    if end > n:
                        end = n
                    sl = end - start
                    if sl <= 0:
                        continue
                    wf = float(w[i, ch])
                    for j in range(sl):
                        x[start + j, ch] += a * wf * float(tpl[i, j])
            else:
                end = t0 + li
                if end > n:
                    end = n
                sl = end - t0
                if sl <= 0:
                    continue
                for ch in range(c):
                    wf = float(w[i, ch])
                    for j in range(sl):
                        x[t0 + j, ch] += a * wf * float(tpl[i, j])

else:  # pragma: no cover

    def scatter_unit_multichannel(*args, **kwargs) -> None:
        raise RuntimeError("numba is not installed")

    def scatter_unit_delayed(*args, **kwargs) -> None:
        raise RuntimeError("numba is not installed")

    def scatter_pool_batched(*args, **kwargs) -> None:
        raise RuntimeError("numba is not installed")


def min_convolve_samples(fs_hz: float) -> int:
    """Heuristic: FFT convolve only pays off once chunk length is long enough."""
    return max(128, int(float(fs_hz) * 0.12))


def superpose_motor_pool_numba(
    x: np.ndarray,
    fs_hz: float,
    rng: np.random.Generator,
    rates_hz: np.ndarray,
    amplitudes_uV: np.ndarray,
    cache,
    *,
    envelope: Optional[np.ndarray] = None,
    time_jitter_std_s: float = 0.0,
    refractory_samples: Optional[np.ndarray] = None,
    spread_across_channels: bool = True,
    spike_indices_fn=None,
) -> int:
    """Numba batched scatter: one JIT call per chunk."""
    if not HAS_NUMBA:
        raise RuntimeError("numba backend requested but numba is not installed")

    from .pool_events import collect_spike_events
    from .pool_fast import superpose_motor_pool_scatter

    n, c = x.shape
    if int(rates_hz.size) == 0:
        return 0

    if not spread_across_channels:
        return superpose_motor_pool_scatter(
            x,
            fs_hz,
            rng,
            rates_hz,
            amplitudes_uV,
            cache,
            envelope=envelope,
            time_jitter_std_s=time_jitter_std_s,
            refractory_samples=refractory_samples,
            spread_across_channels=False,
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
        has_delays = 0
    else:
        dly_pad = np.asarray(dly, dtype=np.int32)
        has_delays = 1

    scatter_pool_batched(
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
