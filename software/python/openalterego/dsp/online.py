"""Online / streaming DSP components.

The offline helpers in :mod:`openalterego.dsp.filters` use filtfilt/sosfiltfilt, which are
great for analysis but not for real-time (they require future samples and add latency).

These classes implement causal filters with internal state so you can run chunk-by-chunk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import signal


@dataclass
class OnlineFilterSpec:
    fs_hz: float
    bandpass_hz: Tuple[float, float] = (1.0, 50.0)
    bandpass_order: int = 4
    notch_hz: Optional[float] = 60.0
    notch_q: float = 30.0


class OnlinePreprocessor:
    """Causal bandpass(+optional notch) + optional rectification + running normalization."""

    def __init__(
        self,
        *,
        fs_hz: float,
        channels: int,
        bandpass_hz: Tuple[float, float] = (1.0, 50.0),
        bandpass_order: int = 4,
        notch_hz: Optional[float] = 60.0,
        notch_q: float = 30.0,
        rectify: bool = False,
        # normalization: exponential moving mean/var
        ema_alpha: float = 0.01,
        eps: float = 1e-6,
    ) -> None:
        if channels <= 0:
            raise ValueError("channels must be > 0")
        if ema_alpha <= 0 or ema_alpha > 1:
            raise ValueError("ema_alpha must be in (0, 1]")

        self.fs_hz = float(fs_hz)
        self.channels = int(channels)
        self.rectify = bool(rectify)
        self.ema_alpha = float(ema_alpha)
        self.eps = float(eps)

        # Bandpass SOS
        self._sos = signal.butter(
            bandpass_order,
            list(bandpass_hz),
            btype="bandpass",
            fs=self.fs_hz,
            output="sos",
        )
        zi = signal.sosfilt_zi(self._sos)  # (sections, 2)
        # expand to (sections, 2, channels)
        self._sos_zi = np.repeat(zi[:, :, None], self.channels, axis=2).astype(np.float32)

        # Notch (optional)
        self._notch = None
        self._notch_zi = None
        if notch_hz is not None:
            b, a = signal.iirnotch(w0=notch_hz, Q=notch_q, fs=self.fs_hz)
            self._notch = (b.astype(np.float64), a.astype(np.float64))
            zi_n = signal.lfilter_zi(b, a)  # (order,)
            self._notch_zi = np.repeat(zi_n[:, None], self.channels, axis=1).astype(np.float32)

        # running normalization state (EMA mean/var)
        self._mu = np.zeros((1, self.channels), dtype=np.float32)
        self._var = np.ones((1, self.channels), dtype=np.float32)

    def reset(self) -> None:
        self._sos_zi[:] = np.repeat(signal.sosfilt_zi(self._sos)[:, :, None], self.channels, axis=2)
        if self._notch is not None and self._notch_zi is not None:
            b, a = self._notch
            zi_n = signal.lfilter_zi(b, a)
            self._notch_zi[:] = np.repeat(zi_n[:, None], self.channels, axis=1)
        self._mu[:] = 0.0
        self._var[:] = 1.0

    def process(self, x: np.ndarray) -> np.ndarray:
        """Process a chunk.

        Parameters
        ----------
        x:
            (time, channels) float array (microvolts by convention).

        Returns
        -------
        y:
            (time, channels) float32 array.
        """
        if x.ndim != 2 or x.shape[1] != self.channels:
            raise ValueError(f"x must have shape (time, {self.channels}), got {x.shape}")

        # Causal bandpass
        y, self._sos_zi = signal.sosfilt(self._sos, x, axis=0, zi=self._sos_zi)

        # Causal notch
        if self._notch is not None and self._notch_zi is not None:
            b, a = self._notch
            y, self._notch_zi = signal.lfilter(b, a, y, axis=0, zi=self._notch_zi)

        if self.rectify:
            y = np.abs(y)

        # Update EMA mean/var
        # We update based on the chunk mean; it's stable and cheap.
        alpha = self.ema_alpha
        chunk_mu = np.mean(y, axis=0, keepdims=True).astype(np.float32)
        self._mu = (1.0 - alpha) * self._mu + alpha * chunk_mu

        # Var of chunk relative to updated mean.
        chunk_var = np.mean((y - self._mu) ** 2, axis=0, keepdims=True).astype(np.float32)
        self._var = (1.0 - alpha) * self._var + alpha * chunk_var

        y = (y - self._mu) / (np.sqrt(self._var) + self.eps)
        return y.astype(np.float32, copy=False)
