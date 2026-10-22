"""Comparable event-window diagnostics for recorded and simulated EMG.

These summaries expose model mismatch; matching them cannot validate anatomy or
counterfactual electrode placements. Use training recordings for calibration.
"""

from __future__ import annotations

import numpy as np


def event_windows(signals: np.ndarray, labels: np.ndarray, *, width: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """One centered, non-overlapping-by-event window for each long label run."""
    x = np.asarray(signals)
    y = np.asarray(labels)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or width < 32:
        raise ValueError("expected matched time-by-channel signals and labels")
    if not np.isfinite(x).all():
        raise ValueError("signals contain non-finite values")
    boundaries = np.r_[0, np.flatnonzero(np.diff(y)) + 1, len(y)]
    windows, classes = [], []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        if stop - start >= width:
            first = (start + stop - width) // 2
            windows.append(x[first:first + width])
            classes.append(y[start])
    if not windows:
        raise ValueError("no label run contains a complete window")
    return np.stack(windows).astype(np.float64), np.asarray(classes)


def signal_signature(windows: np.ndarray, labels: np.ndarray, *, fs_hz: int = 500) -> dict:
    """Summarize filtered 14-channel 500 Hz event windows without fitting labels."""
    x, y = np.asarray(windows, dtype=np.float64), np.asarray(labels)
    if x.ndim != 3 or x.shape[1] < 32 or x.shape[2] != 14 or len(y) != len(x) or fs_hz != 500:
        raise ValueError("expected matching event-by-time-by-14-channel windows at 500 Hz")
    if not np.isfinite(x).all() or not np.any(y == 0) or not np.any(y != 0):
        raise ValueError("finite rest and command windows are required")
    centered = x - np.mean(x, axis=1, keepdims=True)
    rms = np.sqrt(np.mean(centered**2, axis=1))
    wl = np.mean(np.abs(np.diff(x, axis=1)), axis=1)
    fft = np.fft.rfft(centered * np.hanning(x.shape[1])[None, :, None], axis=1)
    power = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(x.shape[1], 1 / fs_hz)
    denominator = np.maximum(np.sum(power[:, (freqs >= 20) & (freqs < 200), :], axis=(1, 2)), 1e-12)
    bands = {f"{lo}_{hi}_hz_fraction": float(np.median(np.sum(power[:, (freqs >= lo) & (freqs < hi), :], axis=(1, 2)) / denominator))
             for lo, hi in ((20, 50), (50, 100), (100, 200))}
    correlations = []
    for window in centered:
        corr = np.corrcoef(window.T)
        finite = np.abs(corr[np.triu_indices(14, 1)])
        correlations.append(float(np.median(finite[np.isfinite(finite)])) if np.isfinite(finite).any() else 0.0)
    rest = float(np.median(rms[y == 0]))
    command = float(np.median(rms[y != 0]))
    return {
        "windows": int(len(y)),
        "rest_windows": int(np.sum(y == 0)),
        "command_windows": int(np.sum(y != 0)),
        "median_centered_rms": float(np.median(rms)),
        "median_waveform_length": float(np.median(wl)),
        "command_to_rest_rms_ratio": command / max(rest, 1e-12),
        "median_absolute_channel_correlation": float(np.median(correlations)),
        "band_power_fractions": bands,
    }
