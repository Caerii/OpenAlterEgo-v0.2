"""Superposition of MUAPs driven by Poisson spikes (independent spike train per synthesis call)."""

from __future__ import annotations

from typing import Optional

import numpy as np


def _normalize_probs(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64).ravel()
    p = np.maximum(p, 0.0)
    s = float(np.sum(p))
    if s <= 1e-12:
        n = p.size
        return np.ones(n, dtype=np.float64) / max(n, 1)
    return p / s


def add_muap_spikes(
    x: np.ndarray,
    fs_hz: float,
    rng: np.random.Generator,
    rate_hz: float,
    channel_weights: np.ndarray,
    muap: np.ndarray,
    amplitude_uV: float,
    *,
    envelope: Optional[np.ndarray] = None,
    time_jitter_std_s: float = 0.0,
    spread_across_channels: bool = False,
    channel_delay_samples: Optional[np.ndarray] = None,
) -> None:
    """Add Poisson MUAP superposition in-place into ``x`` (time, channels), float32.

    If ``spread_across_channels`` is False (legacy): each spike picks **one** channel from
    ``channel_weights`` (multinomial).

    If True (sEMG-like): each spike adds the same temporal MUAP shape to **all** channels with
    amplitudes proportional to ``channel_weights`` (row-normalized pickup / crude volume path).

    If ``envelope`` is length ``n``, scale each spike amplitude by ``envelope[idx]`` (burst shaping).
    ``time_jitter_std_s`` adds Gaussian jitter to each spike time before discretizing to samples.
    ``channel_delay_samples`` optional length-``c`` int vector: per-electrode sample delays when spreading.
    """
    n, c = x.shape
    if n <= 0 or c <= 0:
        return
    probs = _normalize_probs(channel_weights)
    if probs.size != c:
        raise ValueError(f"channel_weights length {probs.size} != x.shape[1] {c}")

    env = np.ones((n,), dtype=np.float32) if envelope is None else np.asarray(envelope, dtype=np.float32).ravel()
    if env.size != n:
        raise ValueError(f"envelope length {env.size} != x.shape[0] {n}")

    T = n / float(fs_hz)
    lam = max(0.0, float(rate_hz)) * T
    n_spikes = int(rng.poisson(lam))
    if n_spikes <= 0:
        return

    spread = bool(spread_across_channels)
    ch_ids = None if spread else rng.choice(c, size=n_spikes, p=probs)
    times_s = rng.uniform(0.0, T, size=n_spikes)
    L = int(muap.shape[0])
    amp0 = float(amplitude_uV)
    m = np.asarray(muap, dtype=np.float32)
    w = probs.astype(np.float32, copy=False)
    dly = None
    if channel_delay_samples is not None:
        dly = np.asarray(channel_delay_samples, dtype=np.int32).ravel()
        if dly.size != c:
            raise ValueError(f"channel_delay_samples length {dly.size} != channels {c}")

    jit = float(time_jitter_std_s)
    for si in range(n_spikes):
        t_s = float(times_s[si])
        t_eff = t_s + (float(rng.normal(0.0, jit)) if jit > 0.0 else 0.0)
        t_eff = float(np.clip(t_eff, 0.0, T * (1.0 - 1e-9)))
        idx = int(t_eff * fs_hz)
        if idx < 0 or idx >= n:
            continue
        amp = amp0 * float(env[min(idx, n - 1)])
        end = min(n, idx + L)
        seg = end - idx
        if seg <= 0:
            continue
        seg_m = m[:seg]
        if spread:
            if dly is not None:
                for ch in range(c):
                    start = idx + int(dly[ch])
                    if start < 0 or start >= n:
                        continue
                    e2 = min(n, start + L)
                    sl = e2 - start
                    if sl <= 0:
                        continue
                    x[start:e2, ch] += (amp * float(w[ch]) * m[:sl]).astype(np.float32, copy=False)
            else:
                x[idx:end, :] += (amp * seg_m[:, None] * w[None, :]).astype(np.float32, copy=False)
        else:
            ci = int(ch_ids[si])
            x[idx:end, ci] += (amp * seg_m).astype(np.float32, copy=False)
