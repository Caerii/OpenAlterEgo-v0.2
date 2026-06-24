"""Partition an event into phone-length runs and align chunk samples to phone indices."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from .durations import phone_duration_weights


def partition_event_to_phones(
    n_samples: int,
    n_phones: int,
    rng: np.random.Generator,
    *,
    min_seg_samples: int = 2,
    duration_weights: Optional[Sequence[float]] = None,
) -> List[int]:
    """Return segment lengths (samples) summing to ``n_samples``, one per phone."""
    n = int(n_phones)
    nt = int(n_samples)
    if n <= 0 or nt <= 0:
        return []
    if n == 1:
        return [nt]
    if duration_weights is not None and len(duration_weights) == n:
        return partition_event_by_weights(
            nt, list(duration_weights), rng, min_seg_samples=min_seg_samples
        )
    if n > nt:
        out = [0] * n
        for i in range(nt):
            out[i % n] += 1
        return [int(x) for x in out]
    ms = max(1, min(int(min_seg_samples), nt // n))
    raw = np.full(n, ms, dtype=np.int64)
    rem = nt - int(np.sum(raw))
    if rem < 0:
        ms = max(1, nt // n)
        raw[:] = ms
        rem = nt - int(np.sum(raw))
    for _ in range(max(0, rem)):
        raw[int(rng.integers(0, n))] += 1
    if int(np.sum(raw)) != nt:
        raise RuntimeError("partition_event_to_phones: internal sum mismatch")
    return [int(x) for x in raw]


def partition_event_by_weights(
    n_samples: int,
    weights: Sequence[float],
    rng: np.random.Generator,
    *,
    min_seg_samples: int = 2,
    jitter: float = 0.08,
) -> List[int]:
    """Duration-weighted partition with small multiplicative jitter per phone."""
    n = len(weights)
    nt = int(n_samples)
    if n <= 0 or nt <= 0:
        return []
    if n == 1:
        return [nt]
    w = np.asarray(weights, dtype=np.float64)
    w = np.maximum(w, 1e-6)
    if float(jitter) > 0:
        w = w * (1.0 + float(jitter) * rng.standard_normal(w.shape))
        w = np.maximum(w, 1e-6)
    ms = max(1, min(int(min_seg_samples), nt // n))
    raw = (w / float(np.sum(w))) * float(nt)
    seg = np.floor(raw).astype(np.int64)
    seg = np.maximum(seg, ms)
    while int(np.sum(seg)) > nt:
        i = int(rng.integers(0, n))
        if seg[i] > ms:
            seg[i] -= 1
    while int(np.sum(seg)) < nt:
        seg[int(rng.integers(0, n))] += 1
    if int(np.sum(seg)) != nt:
        seg[-1] += nt - int(np.sum(seg))
    return [int(x) for x in seg]


def partition_phones_in_event(
    n_samples: int,
    phones: Sequence[str],
    rng: np.random.Generator,
    *,
    min_seg_samples: int = 2,
    duration_weights: Optional[Sequence[float]] = None,
) -> List[int]:
    """Partition a word event using phone list + optional per-phone duration weights."""
    seq = list(phones)
    if not seq:
        return []
    weights = list(duration_weights) if duration_weights is not None else phone_duration_weights(seq)
    return partition_event_to_phones(
        int(n_samples),
        len(seq),
        rng,
        min_seg_samples=min_seg_samples,
        duration_weights=weights,
    )


def iter_phone_slices(
    event_offset: int,
    n_inject: int,
    seg_lens: Sequence[int],
) -> List[Tuple[int, int, int]]:
    """Map injected samples to ``(local_start, local_end, phone_index)`` within ``x[:n_inject]``."""
    if n_inject <= 0 or not seg_lens:
        return []
    cum = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(np.asarray(seg_lens, dtype=np.int64))))
    end_off = int(event_offset) + int(n_inject)
    out: List[Tuple[int, int, int]] = []
    cur = int(event_offset)
    guard = 0
    while cur < end_off and guard < n_inject + len(seg_lens) + 8:
        guard += 1
        pid = int(np.searchsorted(cum, cur, side="right") - 1)
        pid = max(0, min(pid, len(seg_lens) - 1))
        seg_end_abs = int(cum[pid + 1])
        r_end = min(end_off, seg_end_abs)
        a = cur - int(event_offset)
        b = r_end - int(event_offset)
        if b > a:
            out.append((a, b, pid))
        if r_end <= cur:
            break
        cur = r_end
    return out
