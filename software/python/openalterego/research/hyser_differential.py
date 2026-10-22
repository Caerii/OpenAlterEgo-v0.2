"""Digital adjacent-contact differential replay on the Hyser dense array.

These are differences of publisher-preprocessed, common-reference recordings.
They do not reproduce the noise behavior of an analog differential front end.
"""

import numpy as np

from openalterego.research.hyser_features import site_coordinates


def adjacent_pairs(site_names: tuple[str, ...]) -> tuple[tuple[int, int], ...]:
    """All horizontal and vertical neighbor pairs within each 8x8 grid."""
    locations = site_coordinates(site_names)
    lookup = {tuple(map(int, point)): index for index, point in enumerate(locations)}
    pairs = []
    for grid in range(4):
        for row in range(8):
            for col in range(8):
                left = lookup[(grid, row, col)]
                for neighbor in ((grid, row, col + 1), (grid, row + 1, col)):
                    if neighbor in lookup:
                        pairs.append((left, lookup[neighbor]))
    if len(pairs) != 448 or len(set(tuple(sorted(pair)) for pair in pairs)) != 448:
        raise ValueError("expected 448 distinct adjacent physical pairs")
    return tuple(pairs)


def differential_trial_features(signal: np.ndarray,
                                pairs: tuple[tuple[int, int], ...]) -> np.ndarray:
    """Interleaved log RMS and log waveform length for digital differences."""
    signal = np.asarray(signal, dtype=np.float32)
    if (signal.ndim != 2 or signal.shape[1] != 256 or len(signal) < 2 or
            not np.isfinite(signal).all() or not pairs or
            any(a < 0 or a >= 256 or b < 0 or b >= 256 or a == b
                for a, b in pairs)):
        raise ValueError("invalid signal or differential pair map")
    left, right = np.asarray(pairs, dtype=np.int16).T
    difference = signal[:, left] - signal[:, right]
    rms = np.sqrt(np.mean(np.square(difference, dtype=np.float64), axis=0))
    waveform = np.mean(np.abs(np.diff(difference, axis=0)), axis=0)
    return np.stack((np.log(np.maximum(rms, 1e-8)),
                     np.log(np.maximum(waveform, 1e-8))),
                    axis=1).reshape(-1).astype(np.float32)
