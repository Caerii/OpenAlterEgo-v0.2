"""Small, explicit class-prototype update for new-session EMG calibration."""

from __future__ import annotations

import numpy as np


def class_centroids(features: np.ndarray, labels: np.ndarray, classes: int = 9) -> np.ndarray:
    if (features.ndim != 2 or labels.shape != (len(features),) or
            not np.isfinite(features).all() or
            not np.array_equal(np.unique(labels), np.arange(classes))):
        raise ValueError("complete finite labeled feature matrix required")
    return np.stack([features[labels == label].mean(axis=0)
                     for label in range(classes)])


def adapt_centroids(prior: np.ndarray, calibration: np.ndarray,
                    labels: np.ndarray, *, pseudocount: float = 5.0) -> np.ndarray:
    """Shrink each new-session class mean toward its prior prototype."""
    if (prior.ndim != 2 or calibration.ndim != 2 or
            calibration.shape[1] != prior.shape[1] or
            labels.shape != (len(calibration),) or pseudocount <= 0 or
            not np.isfinite(prior).all() or not np.isfinite(calibration).all() or
            any(label < 0 or label >= len(prior) for label in labels)):
        raise ValueError("invalid prior or calibration features")
    updated = prior.copy()
    for label in range(len(prior)):
        row = calibration[labels == label]
        if len(row):
            updated[label] = (pseudocount * prior[label] + row.sum(axis=0)) / (
                pseudocount + len(row))
    return updated


def predict_nearest(features: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    if (features.ndim != 2 or centroids.ndim != 2 or
            features.shape[1] != centroids.shape[1]):
        raise ValueError("feature and centroid dimensions differ")
    distances = ((features[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return distances.argmin(axis=1)
