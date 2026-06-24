"""Signal quality monitoring and motion artifact detection.

Based on research findings:
- Tang et al. (2024): Motion artifacts reduce SNR from 18.9 dB (static) to 12.7 dB (motion)
- Motion causes low-frequency drift (< 5 Hz) and baseline wandering
- High-pass filtering (< 20 Hz) effectively suppresses artifacts
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy import signal


@dataclass
class SignalQualityMetrics:
    """Signal quality metrics for a signal segment.
    
    Attributes
    ----------
    snr_db:
        Signal-to-noise ratio in dB (None if cannot be computed)
    motion_index:
        Motion artifact index (0-1, higher = more motion artifacts)
        Computed from low-frequency drift (< 5 Hz)
    baseline_wander:
        Baseline wandering metric (RMS of low-frequency component)
    signal_power:
        Signal power in the signal band (RMS)
    noise_power:
        Noise power in the noise band (RMS)
    snr_db_per_channel:
        Per-channel SNR (dB) when ``assess_signal_quality(..., per_channel=True)``; else None.
    motion_index_per_channel:
        Per-channel motion index when ``per_channel=True``; else None.
    """
    snr_db: Optional[float] = None
    motion_index: float = 0.0
    baseline_wander: float = 0.0
    signal_power: float = 0.0
    noise_power: float = 0.0
    snr_db_per_channel: Optional[np.ndarray] = None
    motion_index_per_channel: Optional[np.ndarray] = None


def _welch_band_mean_power(
    x: np.ndarray,
    fs_hz: float,
    signal_band_hz: tuple[float, float],
    noise_band_hz: tuple[float, float],
    axis: int = 0,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Return (signal_power_per_ch, noise_power_per_ch, freqs) or (None, None, None) if not computable."""
    if x.size == 0:
        return None, None, None
    nperseg = min(256, int(x.shape[axis]) // 4)
    if nperseg < 4:
        return None, None, None

    if x.ndim == 1:
        freqs, psd = signal.welch(x, fs=fs_hz, nperseg=nperseg, axis=axis)
        psd = psd[None, :]
    else:
        freqs, psd = signal.welch(x, fs=fs_hz, nperseg=nperseg, axis=axis)
        if axis == 0:
            psd = psd.T

    signal_mask = (freqs >= signal_band_hz[0]) & (freqs <= signal_band_hz[1])
    noise_mask = (freqs >= noise_band_hz[0]) & (freqs <= noise_band_hz[1])

    signal_power = np.mean(psd[:, signal_mask], axis=1)
    noise_power = np.mean(psd[:, noise_mask], axis=1)
    noise_power = np.maximum(noise_power, 1e-12)
    return signal_power, noise_power, freqs


def compute_snr_per_channel_db(
    x: np.ndarray,
    fs_hz: float,
    signal_band_hz: tuple[float, float] = (20.0, 450.0),
    noise_band_hz: tuple[float, float] = (0.5, 5.0),
    axis: int = 0,
) -> np.ndarray:
    """Per-channel SNR in dB (Welch band power ratio); -inf where undefined."""
    sig, noise, _f = _welch_band_mean_power(x, fs_hz, signal_band_hz, noise_band_hz, axis=axis)
    if sig is None:
        return np.array([-np.inf], dtype=np.float64)
    ratio = sig / noise
    ratio = np.maximum(ratio, 1e-12)
    return (10.0 * np.log10(ratio)).astype(np.float64)


def weak_channel_indices(
    snr_db_per_channel: np.ndarray,
    *,
    deficit_db: float = 6.0,
) -> List[int]:
    """Channels whose SNR is more than ``deficit_db`` below the median (finite channels only)."""
    s = np.asarray(snr_db_per_channel, dtype=np.float64).ravel()
    finite = s[np.isfinite(s)]
    if finite.size == 0:
        return []
    med = float(np.median(finite))
    out: List[int] = []
    for i, v in enumerate(s):
        if np.isfinite(v) and float(v) < med - float(deficit_db):
            out.append(int(i))
    return out


def compute_snr(
    x: np.ndarray,
    fs_hz: float,
    signal_band_hz: tuple[float, float] = (20.0, 450.0),
    noise_band_hz: tuple[float, float] = (0.5, 5.0),
    axis: int = 0,
) -> float:
    """Compute signal-to-noise ratio in dB.
    
    SNR is computed as the ratio of signal power in the signal band
    to noise power in the noise band.
    
    Parameters
    ----------
    x:
        Input signal (time, channels) or (time,)
    fs_hz:
        Sampling rate
    signal_band_hz:
        Frequency band for signal (default: 20-450 Hz for EMG)
    noise_band_hz:
        Frequency band for noise (default: 0.5-5 Hz for low-frequency artifacts)
    axis:
        Time axis (default: 0)
        
    Returns
    -------
    snr_db:
        SNR in dB, or -np.inf if noise power is zero
    """
    sig, noise, _f = _welch_band_mean_power(x, fs_hz, signal_band_hz, noise_band_hz, axis=axis)
    if sig is None:
        return -np.inf
    snr_linear = sig / noise
    snr_db = 10 * np.log10(np.mean(snr_linear))
    return float(snr_db)


def _motion_per_channel_arrays(
    x: np.ndarray,
    fs_hz: float,
    low_freq_cutoff_hz: float = 5.0,
    axis: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (motion_index per channel, baseline_wander per channel)."""
    if x.size == 0:
        z = np.zeros(1, dtype=np.float64)
        return z, z
    nyquist = fs_hz / 2.0
    if low_freq_cutoff_hz >= nyquist:
        z = np.zeros(x.shape[1] if x.ndim == 2 else 1, dtype=np.float64)
        return z, z

    sos = signal.butter(4, low_freq_cutoff_hz, btype="low", fs=fs_hz, output="sos")
    low_freq = signal.sosfilt(sos, x, axis=axis)

    if x.ndim == 1:
        baseline_per = np.array([float(np.sqrt(np.mean(low_freq**2)))], dtype=np.float64)
        total_per = np.array([float(np.sqrt(np.mean(x**2)))], dtype=np.float64)
    else:
        baseline_per = np.sqrt(np.mean(low_freq**2, axis=axis)).astype(np.float64)
        total_per = np.sqrt(np.mean(x**2, axis=axis)).astype(np.float64)

    motion_per = np.minimum(1.0, baseline_per / (total_per + 1e-6))
    return motion_per, baseline_per


def fast_chunk_motion_index(x: np.ndarray, fs_hz: float, *, lf_cutoff_hz: float = 5.0) -> float:
    """Cheap motion proxy for one chunk (0–1); suitable for online gating."""
    if x.size == 0:
        return 0.0
    xm = x if x.ndim == 1 else np.mean(x.astype(np.float64), axis=1)
    if xm.size < 4:
        return 0.0
    nyq = float(fs_hz) / 2.0
    if lf_cutoff_hz >= nyq:
        return 0.0
    sos = signal.butter(2, lf_cutoff_hz, btype="low", fs=float(fs_hz), output="sos")
    low = signal.sosfilt(sos, xm.astype(np.float64))
    total = float(np.sqrt(np.mean(xm.astype(np.float64) ** 2)) + 1e-9)
    wander = float(np.sqrt(np.mean(low**2)))
    return float(min(1.0, wander / total))


def detect_motion_artifacts(
    x: np.ndarray,
    fs_hz: float,
    low_freq_cutoff_hz: float = 5.0,
    axis: int = 0,
) -> tuple[float, float]:
    """Detect motion artifacts in signal.
    
    Motion artifacts manifest as:
    1. Low-frequency drift (< 5 Hz)
    2. Baseline wandering
    
    Parameters
    ----------
    x:
        Input signal (time, channels) or (time,)
    fs_hz:
        Sampling rate
    low_freq_cutoff_hz:
        Frequency cutoff for low-frequency component (default: 5 Hz)
    axis:
        Time axis (default: 0)
        
    Returns
    -------
    motion_index:
        Motion artifact index (0-1, higher = more motion)
        Computed as normalized low-frequency power
    baseline_wander:
        Baseline wandering metric (RMS of low-frequency component)
    """
    motion_per, baseline_per = _motion_per_channel_arrays(x, fs_hz, low_freq_cutoff_hz, axis=axis)
    return float(np.mean(motion_per)), float(np.mean(baseline_per))


def assess_signal_quality(
    x: np.ndarray,
    fs_hz: float,
    signal_band_hz: tuple[float, float] = (20.0, 450.0),
    noise_band_hz: tuple[float, float] = (0.5, 5.0),
    low_freq_cutoff_hz: float = 5.0,
    axis: int = 0,
    *,
    per_channel: bool = False,
) -> SignalQualityMetrics:
    """Comprehensive signal quality assessment.
    
    Computes SNR, motion artifact index, and baseline wandering metrics.
    
    Parameters
    ----------
    x:
        Input signal (time, channels) or (time,)
    fs_hz:
        Sampling rate
    signal_band_hz:
        Frequency band for signal (default: 20-450 Hz for EMG)
    noise_band_hz:
        Frequency band for noise (default: 0.5-5 Hz)
    low_freq_cutoff_hz:
        Frequency cutoff for motion detection (default: 5 Hz)
    axis:
        Time axis (default: 0)
        
    Returns
    -------
    metrics:
        SignalQualityMetrics with all computed metrics
    """
    # Compute SNR
    snr_db = compute_snr(
        x, fs_hz, signal_band_hz, noise_band_hz, axis=axis
    )
    
    # Detect motion artifacts
    motion_index, baseline_wander = detect_motion_artifacts(
        x, fs_hz, low_freq_cutoff_hz, axis=axis
    )

    snr_pc: Optional[np.ndarray] = None
    mot_pc: Optional[np.ndarray] = None
    if per_channel:
        snr_pc = compute_snr_per_channel_db(x, fs_hz, signal_band_hz, noise_band_hz, axis=axis)
        mot_pc, _b = _motion_per_channel_arrays(x, fs_hz, low_freq_cutoff_hz, axis=axis)
    
    # Compute signal and noise power for reference
    # Use RMS in frequency bands
    if x.size > 0:
        # Simple RMS-based power estimation
        if x.ndim == 1:
            signal_power = float(np.sqrt(np.mean(x ** 2)))
        else:
            signal_power_per_channel = np.sqrt(np.mean(x ** 2, axis=axis))
            signal_power = float(np.mean(signal_power_per_channel))
        noise_power = signal_power * (10 ** (-snr_db / 10.0)) if snr_db > -np.inf else 0.0
    else:
        signal_power = 0.0
        noise_power = 0.0
    
    return SignalQualityMetrics(
        snr_db=snr_db if snr_db > -np.inf else None,
        motion_index=motion_index,
        baseline_wander=baseline_wander,
        signal_power=signal_power,
        noise_power=noise_power,
        snr_db_per_channel=snr_pc,
        motion_index_per_channel=mot_pc,
    )


class OnlineQualityMonitor:
    """Online signal quality monitoring with sliding window.
    
    Maintains a sliding window of signal quality metrics for real-time
    monitoring and alerting.
    """
    
    def __init__(
        self,
        fs_hz: float,
        window_samples: int = 1000,
        signal_band_hz: tuple[float, float] = (20.0, 450.0),
        noise_band_hz: tuple[float, float] = (0.5, 5.0),
        low_freq_cutoff_hz: float = 5.0,
        per_channel: bool = False,
    ) -> None:
        """Initialize online quality monitor.
        
        Parameters
        ----------
        fs_hz:
            Sampling rate
        window_samples:
            Number of samples in sliding window (default: 1000 = 4s at 250 Hz)
        signal_band_hz:
            Frequency band for signal
        noise_band_hz:
            Frequency band for noise
        low_freq_cutoff_hz:
            Frequency cutoff for motion detection
        per_channel:
            If True, fill ``snr_db_per_channel`` / ``motion_index_per_channel`` on updates (slightly more work).
        """
        self.fs_hz = float(fs_hz)
        self.window_samples = int(window_samples)
        self.signal_band_hz = signal_band_hz
        self.noise_band_hz = noise_band_hz
        self.low_freq_cutoff_hz = float(low_freq_cutoff_hz)
        self.per_channel = bool(per_channel)
        
        # Sliding window buffer
        self._buffer: Optional[np.ndarray] = None
        self._buffer_idx = 0
        self._channels: Optional[int] = None
    
    def update(self, x: np.ndarray) -> SignalQualityMetrics:
        """Update monitor with new signal chunk and return current metrics.
        
        Parameters
        ----------
        x:
            Signal chunk (time, channels) or (time,)
            
        Returns
        -------
        metrics:
            Current signal quality metrics
        """
        if x.size == 0:
            return SignalQualityMetrics()
        
        # Normalize to 2D: (time, channels)
        if x.ndim == 1:
            x_2d = x[:, None]  # Add channel dimension
            channels = 1
        else:
            x_2d = x
            channels = x.shape[1]
        
        # Initialize buffer on first call
        if self._buffer is None:
            self._channels = channels
            self._buffer = np.zeros((self.window_samples, self._channels), dtype=np.float32)
        
        # Append new data to buffer (circular buffer)
        chunk_size = x_2d.shape[0]
        for i in range(chunk_size):
            self._buffer[self._buffer_idx, :] = x_2d[i, :]
            self._buffer_idx = (self._buffer_idx + 1) % self.window_samples
        
        # Compute metrics on current buffer
        # Use the most recent data (may wrap around)
        if self._buffer_idx == 0:
            # Buffer is full, use entire buffer
            window_data = self._buffer
        else:
            # Buffer partially filled, use from start
            window_data = self._buffer[:self._buffer_idx]
        
        if window_data.ndim == 2 and window_data.shape[1] == 1:
            window_data = window_data[:, 0]  # Remove singleton channel dim
        
        return assess_signal_quality(
            window_data,
            self.fs_hz,
            self.signal_band_hz,
            self.noise_band_hz,
            self.low_freq_cutoff_hz,
            axis=0,
            per_channel=self.per_channel,
        )
    
    def reset(self) -> None:
        """Reset the monitor (clear buffer)."""
        self._buffer = None
        self._buffer_idx = 0
        self._channels = None
