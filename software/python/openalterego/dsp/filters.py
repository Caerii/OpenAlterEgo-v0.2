"""DSP helpers for OpenAlterEgo.

These are intentionally simple and readable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class FilterSpec:
    fs_hz: float
    bandpass_hz: Tuple[float, float] = (1.0, 50.0)
    bandpass_order: int = 4
    notch_hz: Optional[float] = 60.0
    notch_q: float = 30.0  # higher = narrower notch


def butter_bandpass(spec: FilterSpec) -> Tuple[np.ndarray, np.ndarray]:
    """Return SOS for a Butterworth bandpass."""
    low, high = spec.bandpass_hz
    sos = signal.butter(
        spec.bandpass_order,
        [low, high],
        btype="bandpass",
        fs=spec.fs_hz,
        output="sos",
    )
    return sos


def apply_sos(x: np.ndarray, sos: np.ndarray) -> np.ndarray:
    """Apply SOS filter along time axis.

    Expected shape: (time, channels) or (time,).
    """
    return signal.sosfiltfilt(sos, x, axis=0)


def notch_iir(spec: FilterSpec) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return IIR notch filter coefficients (b, a) or None."""
    if spec.notch_hz is None:
        return None
    b, a = signal.iirnotch(w0=spec.notch_hz, Q=spec.notch_q, fs=spec.fs_hz)
    return b, a


def apply_notch(x: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    return signal.filtfilt(b, a, x, axis=0)


def rectify(x: np.ndarray) -> np.ndarray:
    return np.abs(x)


def normalize(
    x: np.ndarray,
    mode: Literal["zscore", "minmax"] = "zscore",
    eps: float = 1e-8,
) -> np.ndarray:
    if mode == "zscore":
        mu = np.mean(x, axis=0, keepdims=True)
        sig = np.std(x, axis=0, keepdims=True)
        return (x - mu) / (sig + eps)
    if mode == "minmax":
        lo = np.min(x, axis=0, keepdims=True)
        hi = np.max(x, axis=0, keepdims=True)
        return (x - lo) / (hi - lo + eps)
    raise ValueError(f"unknown mode: {mode}")


def preprocess_basic(
    x: np.ndarray,
    fs_hz: float,
    bandpass_hz: Tuple[float, float] = (1.0, 50.0),
    notch_hz: Optional[float] = 60.0,
    rectify_signals: bool = False,
    normalize_mode: Literal["zscore", "minmax"] = "zscore",
) -> np.ndarray:
    """A pragmatic default preprocessing pipeline.

    - bandpass
    - notch
    - optional rectification
    - normalization
    """
    spec = FilterSpec(fs_hz=fs_hz, bandpass_hz=bandpass_hz, notch_hz=notch_hz)
    sos = butter_bandpass(spec)
    y = apply_sos(x, sos)
    notch = notch_iir(spec)
    if notch is not None:
        b, a = notch
        y = apply_notch(y, b, a)
    if rectify_signals:
        y = rectify(y)
    y = normalize(y, mode=normalize_mode)
    return y


def preprocess_streaming(
    x: np.ndarray,
    *,
    fs_hz: float,
    channels: Optional[int] = None,
    bandpass_hz: Tuple[float, float] = (1.0, 50.0),
    notch_hz: Optional[float] = 60.0,
    rectify_signals: bool = False,
    ema_alpha: float = 0.01,
    chunk_samples: int = 128,
) -> np.ndarray:
    """Streaming-compatible preprocessing (causal).

    This runs :class:`openalterego.dsp.online.OnlinePreprocessor` across the full array in chunks
    so it matches realtime behavior.

    It's handy when you want your training preprocessing to be closer to what you'll do online.
    """
    from .online import OnlinePreprocessor

    if x.ndim != 2:
        raise ValueError(f"x must have shape (time, channels), got {x.shape}")
    if channels is None:
        channels = int(x.shape[1])
    if x.shape[1] != channels:
        raise ValueError(f"channel mismatch: x has {x.shape[1]}, channels={channels}")
    if chunk_samples <= 0:
        raise ValueError("chunk_samples must be >0")

    pp = OnlinePreprocessor(
        fs_hz=float(fs_hz),
        channels=int(channels),
        bandpass_hz=bandpass_hz,
        notch_hz=notch_hz,
        rectify=bool(rectify_signals),
        ema_alpha=float(ema_alpha),
    )

    out = np.zeros_like(x, dtype=np.float32)
    for i in range(0, x.shape[0], chunk_samples):
        block = x[i : i + chunk_samples, :]
        out[i : i + block.shape[0], :] = pp.process(block)
    return out
