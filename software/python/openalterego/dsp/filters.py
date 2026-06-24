"""DSP helpers for OpenAlterEgo.

These are intentionally simple and readable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Tuple

import numpy as np
from scipy import signal


# Preprocessing mode type (bandpass presets + pipeline aliases)
PreprocessingMode = Literal["standard", "clinical", "wide", "gowda", "offline", "streaming", "none"]


@dataclass(frozen=True)
class FilterSpec:
    fs_hz: float
    bandpass_hz: Tuple[float, float] = (1.0, 50.0)
    bandpass_order: int = 4
    notch_hz: Optional[float] = 60.0
    notch_q: float = 30.0  # higher = narrower notch
    notch_harmonics: bool = False  # Enable harmonic notching (2nd, 3rd harmonics)


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


def apply_notch_with_harmonics(x: np.ndarray, spec: FilterSpec) -> np.ndarray:
    """Apply notch filter, optionally including harmonics.
    
    Parameters
    ----------
    x:
        Input signal (time, channels) or (time,)
    spec:
        FilterSpec with notch configuration
        
    Returns
    -------
    y:
        Filtered signal with notch(es) applied
    """
    y = x
    if spec.notch_hz is None:
        return y
    
    # Notch fundamental frequency
    notch = notch_iir(spec)
    if notch is not None:
        b, a = notch
        y = apply_notch(y, b, a)
    
    # Notch harmonics if enabled
    if spec.notch_harmonics:
        for harmonic in [2, 3]:  # 2nd and 3rd harmonics
            freq = spec.notch_hz * harmonic
            # Only notch if below Nyquist
            if freq < spec.fs_hz / 2:
                harmonic_spec = FilterSpec(
                    fs_hz=spec.fs_hz,
                    bandpass_hz=spec.bandpass_hz,
                    bandpass_order=spec.bandpass_order,
                    notch_hz=freq,
                    notch_q=spec.notch_q,
                    notch_harmonics=False,  # Don't recurse
                )
                harmonic_notch = notch_iir(harmonic_spec)
                if harmonic_notch is not None:
                    b, a = harmonic_notch
                    y = apply_notch(y, b, a)
    
    return y


def get_filter_spec_for_mode(
    mode: PreprocessingMode,
    fs_hz: float,
    notch_hz: Optional[float] = None,
    notch_harmonics: bool = False,
) -> FilterSpec:
    """Get FilterSpec for a preprocessing mode.
    
    Parameters
    ----------
    mode:
        Preprocessing mode:
        - "standard" (1-50 Hz): Classic AlterEgo range for silent speech envelope
        - "clinical" (0.5-8 Hz): Narrow band for dysphonic/MS patients
        - "wide" (20-450 Hz): Modern EMG range based on recent literature (2021-2024)
          Note: Requires fs_hz >= 920 Hz (Nyquist > 450 Hz).
        - "gowda" (80-1000 Hz): Gowda 2025 / emg2speech paper bandpass (3rd-order Butterworth)
          Requires fs_hz >= 2010 Hz.
        - "offline", "streaming", "none": Use standard bandpass
    fs_hz:
        Sampling rate
    notch_hz:
        Notch frequency (None to disable, or 50/60 for power line)
    notch_harmonics:
        Whether to notch harmonics (2nd, 3rd) of notch_hz
        
    Returns
    -------
    FilterSpec configured for the mode
    """
    if mode == "clinical":
        return FilterSpec(
            fs_hz=fs_hz,
            bandpass_hz=(0.5, 8.0),
            bandpass_order=4,
            notch_hz=notch_hz,
            notch_q=30.0,
            notch_harmonics=notch_harmonics,
        )
    elif mode == "wide":
        nyquist = fs_hz / 2.0
        if nyquist <= 450.0 + 10.0:
            raise ValueError(
                f"Wide mode requires fs_hz >= 920 Hz (Nyquist > 460 Hz for 450 Hz + 10 Hz margin). "
                f"Got fs_hz={fs_hz} (Nyquist={nyquist} Hz). "
                f"Use 'standard' or 'clinical' mode for lower sampling rates."
            )
        return FilterSpec(
            fs_hz=fs_hz,
            bandpass_hz=(20.0, 450.0),
            bandpass_order=4,
            notch_hz=notch_hz,
            notch_q=30.0,
            notch_harmonics=notch_harmonics,
        )
    elif mode == "gowda":
        nyquist = fs_hz / 2.0
        if nyquist <= 1000.0 + 10.0:
            raise ValueError(
                f"Gowda mode requires fs_hz >= 2010 Hz (Nyquist > 1010 Hz for 1000 Hz + margin). "
                f"Got fs_hz={fs_hz} (Nyquist={nyquist} Hz)."
            )
        return FilterSpec(
            fs_hz=fs_hz,
            bandpass_hz=(80.0, 1000.0),
            bandpass_order=3,
            notch_hz=notch_hz,
            notch_q=30.0,
            notch_harmonics=notch_harmonics,
        )
    else:  # standard, offline, streaming, none
        return FilterSpec(
            fs_hz=fs_hz,
            bandpass_hz=(1.0, 50.0),
            bandpass_order=4,
            notch_hz=notch_hz,
            notch_q=30.0,
            notch_harmonics=notch_harmonics,
        )


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
    notch_harmonics: bool = False,
    rectify_signals: bool = False,
    normalize_mode: Literal["zscore", "minmax"] = "zscore",
    mode: Optional[PreprocessingMode] = None,
) -> np.ndarray:
    """A pragmatic default preprocessing pipeline.

    - bandpass
    - notch (optionally with harmonics)
    - optional rectification
    - normalization
    
    Parameters
    ----------
    x:
        Input signal (time, channels)
    fs_hz:
        Sampling rate
    bandpass_hz:
        Bandpass frequency range (overridden if mode is provided)
    notch_hz:
        Notch frequency (None to disable, 50/60 for power line)
    notch_harmonics:
        Whether to notch harmonics of notch_hz
    rectify_signals:
        Whether to rectify (abs) the signal
    normalize_mode:
        Normalization mode: "zscore" or "minmax"
    mode:
        Preprocessing mode: "standard" (1-50 Hz), "clinical" (0.5-8 Hz), or "wide" (20-450 Hz)
        If provided, overrides bandpass_hz
    """
    # Use mode-based spec if provided
    if mode is not None:
        spec = get_filter_spec_for_mode(mode, fs_hz, notch_hz, notch_harmonics)
    else:
        spec = FilterSpec(
            fs_hz=fs_hz,
            bandpass_hz=bandpass_hz,
            notch_hz=notch_hz,
            notch_harmonics=notch_harmonics,
        )
    
    sos = butter_bandpass(spec)
    y = apply_sos(x, sos)
    
    # Apply notch with optional harmonics
    y = apply_notch_with_harmonics(y, spec)
    
    if rectify_signals:
        y = rectify(y)
    y = normalize(y, mode=normalize_mode)
    # Cast to float32 to match input dtype
    return y.astype(np.float32, copy=False)


def preprocess_streaming(
    x: np.ndarray,
    *,
    fs_hz: float,
    channels: Optional[int] = None,
    bandpass_hz: Tuple[float, float] = (1.0, 50.0),
    notch_hz: Optional[float] = 60.0,
    notch_harmonics: bool = False,
    rectify_signals: bool = False,
    ema_alpha: float = 0.01,
    chunk_samples: int = 128,
    mode: Optional[PreprocessingMode] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> np.ndarray:
    """Streaming-compatible preprocessing (causal).

    This runs :class:`openalterego.dsp.online.OnlinePreprocessor` across the full array in chunks
    so it matches realtime behavior.

    It's handy when you want your training preprocessing to be closer to what you'll do online.
    
    Parameters
    ----------
    mode:
        Preprocessing mode: "standard" (1-50 Hz), "clinical" (0.5-8 Hz), or "wide" (20-450 Hz)
        If provided, overrides bandpass_hz
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

    # Use mode-based bandpass if provided
    if mode == "clinical":
        bandpass_hz = (0.5, 8.0)
    elif mode == "wide":
        from .emg_config import validate_emg_wide_fs

        validate_emg_wide_fs(float(fs_hz))
        bandpass_hz = (20.0, 450.0)
    elif mode == "gowda":
        from .emg_config import validate_emg_gowda_fs

        validate_emg_gowda_fs(float(fs_hz))
        bandpass_hz = (80.0, 1000.0)
    elif mode == "standard" or mode is None:
        pass  # Use provided bandpass_hz
    # Note: "offline", "streaming", "none" modes are handled by caller

    pp = OnlinePreprocessor(
        fs_hz=float(fs_hz),
        channels=int(channels),
        bandpass_hz=bandpass_hz,
        notch_hz=notch_hz,
        notch_harmonics=notch_harmonics,
        rectify=bool(rectify_signals),
        ema_alpha=float(ema_alpha),
    )

    out = np.zeros_like(x, dtype=np.float32)
    n_samples = int(x.shape[0])
    n_chunks = max(1, (n_samples + int(chunk_samples) - 1) // int(chunk_samples))
    chunk_i = 0
    for i in range(0, n_samples, chunk_samples):
        block = x[i : i + chunk_samples, :]
        out[i : i + block.shape[0], :] = pp.process(block)
        if progress_cb is not None:
            progress_cb(chunk_i, n_chunks)
        chunk_i += 1
    return out
