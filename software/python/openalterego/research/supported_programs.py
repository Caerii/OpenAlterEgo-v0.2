"""Batched ridge sufficient statistics with labeled new-session support."""

import numpy as np

from openalterego.research.dense_programs import DenseFold


def prepare_supported_fold(day1: tuple[np.ndarray, np.ndarray],
                           support: tuple[np.ndarray, np.ndarray],
                           query: tuple[np.ndarray, np.ndarray], *,
                           features_per_site: int,
                           support_weight: float = 1.0) -> DenseFold:
    """Fit statistics on day 1 plus support; preserve query for scoring only."""
    x, y = day1
    xs, ys = support
    xq, yq = query
    x = np.asarray(x, dtype=np.float64)
    xs = np.asarray(xs, dtype=np.float64)
    xq = np.asarray(xq, dtype=np.float64)
    y, ys, yq = np.asarray(y), np.asarray(ys), np.asarray(yq)
    if (features_per_site < 1 or support_weight < 0 or
            not np.isfinite(support_weight) or x.ndim != 2 or
            xs.ndim != 2 or xq.ndim != 2 or not len(x) or not len(xs) or
            not len(xq) or x.shape[1] % features_per_site or
            xs.shape[1] != x.shape[1] or xq.shape[1] != x.shape[1] or
            len(y) != len(x) or len(ys) != len(xs) or len(yq) != len(xq) or
            not all(np.isfinite(value).all() for value in (x, xs, xq))):
        raise ValueError("invalid day-one, support or query arrays")
    classes = np.unique(np.r_[y, ys] if support_weight > 0 else y)
    day1_ids = np.searchsorted(classes, y)
    counts = np.bincount(day1_ids, minlength=len(classes))
    per_class_mass = len(y) / len(np.unique(y))
    day1_weights = per_class_mass / counts[day1_ids]
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1e-6)
    baseline = np.column_stack(((x - mean) / scale, np.ones(len(x))))
    if support_weight > 0:
        support_ids = np.searchsorted(classes, ys)
        support_counts = np.bincount(support_ids, minlength=len(classes))
        support_weights = support_weight * per_class_mass / support_counts[support_ids]
        design = np.vstack((baseline, np.column_stack(
            ((xs - mean) / scale, np.ones(len(xs))))))
        ids = np.r_[day1_ids, support_ids]
        weights = np.r_[day1_weights, support_weights]
    else:
        design, ids, weights = baseline, day1_ids, day1_weights
    query_design = np.column_stack(((xq - mean) / scale, np.ones(len(xq))))
    onehot = np.eye(len(classes))[ids]
    return DenseFold(design.T @ (design * weights[:, None]),
                     design.T @ (onehot * weights[:, None]),
                     query_design, yq, classes)
