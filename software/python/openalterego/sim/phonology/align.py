"""Phone alignment modes for within-word pseudo / corpus-duration segmentation."""

from __future__ import annotations

from typing import List, Literal, Optional, Sequence

import numpy as np

from .durations import phone_duration_weight, phone_duration_weights
from .timeline import partition_event_by_weights, partition_phones_in_event

AlignMode = Literal["pseudo", "corpus_duration"]

# TTS-informed mean phone durations (ms) from English speaking-rate corpora (approximate).
_PHONE_DURATION_MS: dict[str, float] = {
    "AA": 95, "AE": 88, "AH": 72, "AO": 98, "AW": 110, "AY": 105,
    "B": 55, "CH": 68, "D": 52, "DH": 58, "EH": 82, "ER": 92, "EY": 95,
    "F": 75, "G": 58, "HH": 50, "IH": 78, "IY": 85, "JH": 65, "K": 55,
    "L": 70, "M": 68, "N": 68, "NG": 72, "OW": 98, "OY": 102, "P": 55,
    "R": 72, "S": 78, "SH": 82, "T": 52, "TH": 62, "UH": 75, "UW": 88,
    "V": 72, "W": 70, "Y": 68, "Z": 75, "ZH": 80,
}


def phone_duration_ms(phone: str) -> float:
    key = str(phone).strip().upper()
    return float(_PHONE_DURATION_MS.get(key, 80.0))


def phone_duration_weights_corpus(phones: Sequence[str]) -> List[float]:
    return [phone_duration_ms(p) for p in phones]


def partition_phones_aligned(
    n_samples: int,
    phones: Sequence[str],
    *,
    mode: AlignMode = "pseudo",
    seed: int = 0,
    min_seg_samples: int = 2,
) -> List[int]:
    """Partition word samples into phone segments using the chosen align mode."""
    seq = list(phones)
    if not seq:
        return []
    n = int(n_samples)
    if str(mode) == "corpus_duration":
        weights = phone_duration_weights_corpus(seq)
        rng = np.random.default_rng(int(seed))
        return partition_event_by_weights(
            n,
            weights,
            rng,
            min_seg_samples=min_seg_samples,
            jitter=0.02,
        )
    weights = phone_duration_weights(seq)
    rng = np.random.default_rng(int(seed))
    return partition_event_by_weights(
        n,
        weights,
        rng,
        min_seg_samples=min_seg_samples,
        jitter=0.08,
    )
