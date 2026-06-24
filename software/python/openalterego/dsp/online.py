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
    notch_harmonics: bool = False  # Enable harmonic notching


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
        notch_harmonics: bool = False,
        rectify: bool = False,
        # normalization: exponential moving mean/var
        ema_alpha: float = 0.01,
        eps: float = 1e-6,
        motion_gate: bool = False,
        motion_threshold: float = 0.35,
        motion_attenuation: float = 0.15,
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
        self.notch_harmonics = bool(notch_harmonics)
        self.motion_gate = bool(motion_gate)
        self.motion_threshold = float(motion_threshold)
        self.motion_attenuation = float(np.clip(motion_attenuation, 0.0, 1.0))
        self.last_motion_index: float = 0.0
        self.last_motion_gated: bool = False

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

        # Notch (optional) - fundamental frequency
        self._notch = None
        self._notch_zi = None
        if notch_hz is not None:
            b, a = signal.iirnotch(w0=notch_hz, Q=notch_q, fs=self.fs_hz)
            self._notch = (b.astype(np.float64), a.astype(np.float64))
            zi_n = signal.lfilter_zi(b, a)  # (order,)
            self._notch_zi = np.repeat(zi_n[:, None], self.channels, axis=1).astype(np.float32)
        
        # Harmonic notches (if enabled)
        self._harmonic_notches: list[tuple[tuple[np.ndarray, np.ndarray], np.ndarray]] = []
        if notch_harmonics and notch_hz is not None:
            for harmonic in [2, 3]:  # 2nd and 3rd harmonics
                freq = notch_hz * harmonic
                if freq < self.fs_hz / 2:  # Below Nyquist
                    b, a = signal.iirnotch(w0=freq, Q=notch_q, fs=self.fs_hz)
                    zi_n = signal.lfilter_zi(b, a)
                    self._harmonic_notches.append(
                        ((b.astype(np.float64), a.astype(np.float64)),
                         np.repeat(zi_n[:, None], self.channels, axis=1).astype(np.float32))
                    )

        # running normalization state (EMA mean/var)
        self._mu = np.zeros((1, self.channels), dtype=np.float32)
        self._var = np.ones((1, self.channels), dtype=np.float32)

    def reset(self) -> None:
        self._sos_zi[:] = np.repeat(signal.sosfilt_zi(self._sos)[:, :, None], self.channels, axis=2)
        if self._notch is not None and self._notch_zi is not None:
            b, a = self._notch
            zi_n = signal.lfilter_zi(b, a)
            self._notch_zi[:] = np.repeat(zi_n[:, None], self.channels, axis=1)
        # Reset harmonic notches
        for i, ((b, a), zi_ref) in enumerate(self._harmonic_notches):
            zi_n = signal.lfilter_zi(b, a)
            self._harmonic_notches[i] = ((b, a), np.repeat(zi_n[:, None], self.channels, axis=1).astype(np.float32))
        self._mu[:] = 0.0
        self._var[:] = 1.0
        self.last_motion_index = 0.0
        self.last_motion_gated = False

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

        from .quality import fast_chunk_motion_index

        self.last_motion_index = fast_chunk_motion_index(x, self.fs_hz)
        self.last_motion_gated = bool(
            self.motion_gate and self.last_motion_index >= self.motion_threshold
        )

        # Causal bandpass
        y, self._sos_zi = signal.sosfilt(self._sos, x, axis=0, zi=self._sos_zi)

        # Causal notch (fundamental)
        if self._notch is not None and self._notch_zi is not None:
            b, a = self._notch
            y, self._notch_zi = signal.lfilter(b, a, y, axis=0, zi=self._notch_zi)
        
        # Causal harmonic notches
        for i, ((b, a), zi_ref) in enumerate(self._harmonic_notches):
            y, new_zi = signal.lfilter(b, a, y, axis=0, zi=zi_ref)
            self._harmonic_notches[i] = ((b, a), new_zi)

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

        if self.last_motion_gated:
            y = y * float(self.motion_attenuation)

        return y.astype(np.float32, copy=False)
