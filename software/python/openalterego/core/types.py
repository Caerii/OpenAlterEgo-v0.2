"""Core datatypes used across OpenAlterEgo.

Keep these simple and dependency-light so both the Python runtime and any future embedded/edge
implementations can mirror them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class FrameChunk:
    """A chunk of multichannel samples.

    Attributes
    ----------
    samples:
        Array of shape (time, channels), float32 in microvolts by convention.
    fs_hz:
        Sampling rate.
    t0:
        Unix timestamp (seconds) for the *first* sample in this chunk (best-effort).
    seq0:
        Monotonic sample sequence index for the first sample in this chunk (optional).
    meta:
        Arbitrary metadata (device name, packet loss counters, etc.)
    """

    samples: np.ndarray
    fs_hz: int
    t0: float
    seq0: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.samples.ndim != 2:
            raise ValueError(f"samples must be 2D (time, channels), got shape={self.samples.shape}")
        if self.samples.dtype not in (np.float32, np.float64):
            object.__setattr__(self, "samples", self.samples.astype(np.float32, copy=False))


@dataclass(frozen=True)
class TokenEvent:
    """A decoded token prediction."""

    token: str
    confidence: float
    t: float
    seq: int = 0
    source: str = "unknown"
    meta: Dict[str, Any] = field(default_factory=dict)
