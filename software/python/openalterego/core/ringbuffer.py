"""A small, fast ring buffer for streaming multichannel time series."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RingBuffer:
    """Ring buffer for arrays shaped (time, channels).

    Notes
    -----
    - Capacity is fixed. Old samples are overwritten when full.
    - `get_last()` returns a *copy* so downstream code can safely mutate.
    """

    capacity: int
    channels: int
    dtype: np.dtype = np.float32

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("capacity must be > 0")
        if self.channels <= 0:
            raise ValueError("channels must be > 0")
        self._buf = np.zeros((self.capacity, self.channels), dtype=self.dtype)
        self._write = 0
        self._filled = 0
        self._total_written = 0  # total frames ever appended

    @property
    def filled(self) -> int:
        return self._filled

    @property
    def total_written(self) -> int:
        return self._total_written

    def append(self, x: np.ndarray) -> None:
        """Append a block of samples shaped (time, channels)."""
        if x.ndim != 2 or x.shape[1] != self.channels:
            raise ValueError(f"x must have shape (time, {self.channels}), got {x.shape}")
        n = int(x.shape[0])
        if n == 0:
            return

        # If the incoming block is bigger than capacity, keep only the last part.
        if n >= self.capacity:
            x = x[-self.capacity :, :]
            n = int(x.shape[0])

        end = self._write + n
        if end <= self.capacity:
            self._buf[self._write : end, :] = x
        else:
            first = self.capacity - self._write
            self._buf[self._write :, :] = x[:first, :]
            self._buf[: end % self.capacity, :] = x[first:, :]

        self._write = end % self.capacity
        self._filled = min(self.capacity, self._filled + n)
        self._total_written += n

    def get_last(self, n: int) -> np.ndarray:
        """Return the most recent n samples, shape (n, channels)."""
        n = int(n)
        if n <= 0:
            raise ValueError("n must be > 0")
        if n > self._filled:
            raise ValueError(f"buffer has only {self._filled} samples, requested {n}")

        if self._filled < self.capacity:
            # Not wrapped yet; data is contiguous from 0..filled.
            start = self._filled - n
            return self._buf[start : self._filled, :].copy()

        # Full buffer; may wrap.
        start = (self._write - n) % self.capacity
        if start < self._write:
            return self._buf[start : start + n, :].copy()

        part1 = self.capacity - start
        a = self._buf[start:, :]
        b = self._buf[: n - part1, :]
        return np.vstack([a, b]).copy()
