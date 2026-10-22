"""Time-resolved channel-local EMG features from fixed-length Hyser trials."""

import numpy as np

from openalterego.research.hyser_features import trial_features


def temporal_trial_features(signal: np.ndarray, *, bins: int = 4) -> np.ndarray:
    """Per-channel whole-trial and ordered-bin log RMS/waveform length.

    A four-bin record has ten features per channel. It retains the earlier
    two-feature whole-trial baseline in columns 0:2 of each channel block.
    """
    if signal.ndim != 2 or signal.shape[1] != 256 or bins < 2 or len(signal) < 2 * bins:
        raise ValueError("invalid trial or temporal bin count")
    chunks = np.array_split(signal, bins, axis=0)
    if any(len(chunk) < 2 for chunk in chunks):
        raise ValueError("temporal bin has fewer than two samples")
    values = [trial_features(signal).reshape(256, 2)]
    values.extend(trial_features(chunk).reshape(256, 2) for chunk in chunks)
    result = np.concatenate(values, axis=1).reshape(-1)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite temporal features")
    return result
