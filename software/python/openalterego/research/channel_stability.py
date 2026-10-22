"""Channel-local class-pattern information and observed session change."""

import numpy as np


def class_contrast(features: np.ndarray, labels: np.ndarray,
                   *, classes: int = 9, channels: int = 14
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Return class-balanced contrasts and within-class feature variance.

    Removing the mean across class centroids avoids calling a shared
    recording-wide amplitude offset a change in command representation.
    """
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    if (features.ndim != 2 or features.shape[1] != 2 * channels or
            labels.shape != (len(features),) or classes < 2 or
            channels < 1 or not np.isfinite(features).all() or
            not np.array_equal(np.unique(labels), np.arange(classes))):
        raise ValueError("invalid channel contrast input")
    shaped = features.reshape(len(features), channels, 2)
    means = np.stack([shaped[labels == label].mean(axis=0)
                      for label in range(classes)])
    within = np.mean([(shaped[labels == label] - means[label]).var(axis=0)
                      for label in range(classes)], axis=0)
    return means - means.mean(axis=0), within
