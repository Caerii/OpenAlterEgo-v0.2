"""Coarticulation: overlapping phone activation envelopes (raised-cosine crossfades)."""

from __future__ import annotations

from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_COARTICULATION_OVERLAP_FRAC = 0.28


def _raised_cosine_ramp_up(n: int) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    t = np.arange(int(n), dtype=np.float64)
    return (0.5 * (1.0 - np.cos(np.pi * t / float(n)))).astype(np.float32)


def _raised_cosine_ramp_down(n: int) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    t = np.arange(int(n), dtype=np.float64)
    return (0.5 * (1.0 + np.cos(np.pi * t / float(n)))).astype(np.float32)


def _boundary_overlap_samples(
    left_len: int,
    right_len: int,
    *,
    overlap_fraction: float,
    min_overlap_samples: int,
    max_overlap_samples: Optional[int],
) -> int:
    lo = int(left_len)
    ro = int(right_len)
    ov = int(round(float(overlap_fraction) * float(min(lo, ro))))
    ov = max(int(min_overlap_samples), ov)
    if max_overlap_samples is not None:
        ov = min(int(max_overlap_samples), ov)
    ov = min(ov, max(1, lo // 2), max(1, ro // 2))
    return max(0, ov)


def build_phone_coarticulation_envelopes(
    seg_lens: Sequence[int],
    *,
    overlap_fraction: float = DEFAULT_COARTICULATION_OVERLAP_FRAC,
    min_overlap_samples: int = 50,
    max_overlap_samples: Optional[int] = None,
) -> np.ndarray:
    """Per-phone activation envelopes over the word, shape ``(n_phones, n_samples)``.

    Each phone's influence extends into neighbors by ``overlap`` samples at boundaries
    (raised-cosine ramps). Columns are normalized so active phones sum to ~1.
    """
    seg = [int(x) for x in seg_lens]
    n_phones = len(seg)
    n_samples = int(sum(seg))
    if n_phones <= 0 or n_samples <= 0:
        return np.zeros((0, 0), dtype=np.float32)
    if n_phones == 1:
        return np.ones((1, n_samples), dtype=np.float32)

    cum = [0]
    for s in seg:
        cum.append(cum[-1] + int(s))

    env = np.zeros((n_phones, n_samples), dtype=np.float32)
    for i in range(n_phones):
        s0, s1 = int(cum[i]), int(cum[i + 1])
        left_ov = (
            _boundary_overlap_samples(
                seg[i - 1],
                seg[i],
                overlap_fraction=overlap_fraction,
                min_overlap_samples=min_overlap_samples,
                max_overlap_samples=max_overlap_samples,
            )
            if i > 0
            else 0
        )
        right_ov = (
            _boundary_overlap_samples(
                seg[i],
                seg[i + 1],
                overlap_fraction=overlap_fraction,
                min_overlap_samples=min_overlap_samples,
                max_overlap_samples=max_overlap_samples,
            )
            if i < n_phones - 1
            else 0
        )
        rs = max(0, s0 - left_ov)
        re = min(n_samples, s1 + right_ov)
        reg_len = int(re - rs)
        if reg_len <= 0:
            continue
        w = np.zeros(reg_len, dtype=np.float32)
        core_a = int(s0 - rs)
        core_b = int(s1 - rs)
        w[core_a:core_b] = 1.0
        if core_a > 0:
            w[:core_a] = _raised_cosine_ramp_up(core_a)
        tail = reg_len - core_b
        if tail > 0:
            w[core_b:] = _raised_cosine_ramp_down(tail)
        env[i, rs:re] = w

    col_sum = np.sum(env, axis=0, keepdims=True)
    dead = col_sum.ravel() <= 1e-8
    if bool(np.any(dead)):
        env[:, dead] = 0.0
        col_sum = np.sum(env, axis=0, keepdims=True)
    col_sum = col_sum + 1e-8
    env /= col_sum
    return env.astype(np.float32, copy=False)


def _contiguous_positive_runs(weights: np.ndarray, *, eps: float = 1e-5) -> List[Tuple[int, int]]:
    w = np.asarray(weights, dtype=np.float32).ravel()
    runs: List[Tuple[int, int]] = []
    i = 0
    n = int(w.size)
    while i < n:
        while i < n and float(w[i]) <= eps:
            i += 1
        if i >= n:
            break
        j = i + 1
        while j < n and float(w[j]) > eps:
            j += 1
        runs.append((i, j))
        i = j
    return runs


def iter_coarticulated_phone_jobs(
    event_offset: int,
    n_inject: int,
    seg_lens: Sequence[int],
    coart_env: np.ndarray,
    *,
    eps: float = 1e-5,
) -> Iterator[Tuple[int, int, int, np.ndarray]]:
    """Yield ``(local_a, local_b, phone_idx, env_slice)`` for motor injection.

    Multiple jobs may cover the same samples (different phones) during overlap.
    """
    if n_inject <= 0 or coart_env.size == 0:
        return
    off = int(event_offset)
    n_ev = int(coart_env.shape[1])
    end = min(off + int(n_inject), n_ev)
    if end <= off:
        return
    n_phones = int(coart_env.shape[0])
    for pid in range(n_phones):
        act = coart_env[pid, off:end]
        for a, b in _contiguous_positive_runs(act, eps=eps):
            yield int(a), int(b), int(pid), act[a:b].copy()


def coarticulation_overlap_ms(
    seg_lens: Sequence[int],
    fs_hz: float,
    *,
    overlap_fraction: float = DEFAULT_COARTICULATION_OVERLAP_FRAC,
    min_overlap_ms: float = 10.0,
) -> List[float]:
    """Overlap duration (ms) at each phone boundary."""
    seg = [int(x) for x in seg_lens]
    fs = float(fs_hz)
    min_samp = max(1, int(round(float(min_overlap_ms) * fs / 1000.0)))
    out: List[float] = []
    for i in range(len(seg) - 1):
        ov = _boundary_overlap_samples(
            seg[i],
            seg[i + 1],
            overlap_fraction=overlap_fraction,
            min_overlap_samples=min_samp,
            max_overlap_samples=None,
        )
        out.append(1000.0 * float(ov) / fs)
    return out
