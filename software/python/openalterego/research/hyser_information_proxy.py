"""Cheap label-aware Gaussian discriminability proxy for recorded EMG sites.

This is a day-one feature statistic, not a biophysical contact model. Its
ability to predict future-session decoding must be measured separately.
"""

import numpy as np


def class_covariances(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return class-balanced between- and within-class feature covariance."""
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    if (features.ndim != 2 or labels.shape != (len(features),) or
            len(features) < 2 or not np.isfinite(features).all()):
        raise ValueError("invalid features or labels")
    classes = np.unique(labels)
    if len(classes) < 2 or any(np.sum(labels == label) < 2 for label in classes):
        raise ValueError("at least two classes with repeated observations required")
    scale = np.maximum(features.std(axis=0), 1e-6)
    normalized = (features - features.mean(axis=0)) / scale
    means = np.stack([normalized[labels == label].mean(axis=0) for label in classes])
    center = means.mean(axis=0)
    between = (means - center).T @ (means - center) / len(classes)
    within = np.mean([
        (normalized[labels == label] - means[index]).T @
        (normalized[labels == label] - means[index]) / np.sum(labels == label)
        for index, label in enumerate(classes)
    ], axis=0)
    return between, within


def score_designs(between: np.ndarray, within: np.ndarray, designs: np.ndarray,
                  *, ridge: float = 0.25, batch_size: int = 2048) -> np.ndarray:
    """Gaussian between/within information score for two features per site."""
    between = np.asarray(between, dtype=np.float64)
    within = np.asarray(within, dtype=np.float64)
    designs = np.asarray(designs, dtype=np.int16)
    if (between.ndim != 2 or between.shape != within.shape or
            between.shape[0] != between.shape[1] or between.shape[0] % 2 or
            designs.ndim != 2 or not 1 <= designs.shape[1] <= between.shape[0] // 2 or
            not len(designs) or np.any(designs < 0) or
            np.any(designs >= between.shape[0] // 2) or
            any(len(set(row)) != len(row) for row in designs) or
            not np.isfinite(ridge) or ridge <= 0 or batch_size < 1):
        raise ValueError("invalid covariance scoring inputs")
    output = np.empty(len(designs), dtype=np.float64)
    for start in range(0, len(designs), batch_size):
        block = designs[start:start + batch_size]
        columns = np.stack((2 * block, 2 * block + 1), axis=-1).reshape(len(block), -1)
        w = within[columns[:, :, None], columns[:, None, :]]
        b = between[columns[:, :, None], columns[:, None, :]]
        w = w + ridge * np.eye(columns.shape[1])
        sign_w, log_w = np.linalg.slogdet(w)
        sign_t, log_t = np.linalg.slogdet(w + b)
        if np.any(sign_w <= 0) or np.any(sign_t <= 0):
            raise ValueError("invalid local covariance")
        output[start:start + len(block)] = (log_t - log_w) / (2 * np.log(2))
    return output
