"""Explicitly labeled new-session support and disjoint query evaluation."""

import numpy as np
from sklearn.metrics import balanced_accuracy_score


def support_query(labels: np.ndarray, shots_per_class: int) -> tuple[np.ndarray, np.ndarray]:
    """First K recorded repetitions per class are support; the rest query."""
    if labels.ndim != 1 or shots_per_class < 1:
        raise ValueError("invalid support split")
    support, query = [], []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        if len(indices) <= shots_per_class:
            raise ValueError("class lacks a held-out query repetition")
        support.extend(indices[:shots_per_class])
        query.extend(indices[shots_per_class:])
    return np.asarray(sorted(support)), np.asarray(sorted(query))


def ridge_query_score(day_one: tuple[np.ndarray, np.ndarray],
                      day_two: tuple[np.ndarray, np.ndarray],
                      channels: list[int], *, features_per_site: int,
                      alpha: float, support: np.ndarray, query: np.ndarray,
                      support_weight: float) -> float:
    x, labels = day_one
    xt, target = day_two
    if (x.ndim != 2 or xt.ndim != 2 or
            x.shape[1] != 256 * features_per_site or
            xt.shape[1] != 256 * features_per_site or
            alpha <= 0 or support_weight < 0 or
            len(set(support) & set(query)) or
            any(index < 0 or index >= len(xt) for index in np.r_[support, query])):
        raise ValueError("invalid few-shot scoring inputs")
    columns = [features_per_site * channel + feature for channel in channels
               for feature in range(features_per_site)]
    mean = x[:, columns].mean(axis=0)
    scale = np.maximum(x[:, columns].std(axis=0), 1e-6)
    baseline = np.column_stack(((x[:, columns] - mean) / scale, np.ones(len(x))))
    support_design = np.column_stack(((xt[support][:, columns] - mean) / scale,
                                      np.ones(len(support))))
    query_design = np.column_stack(((xt[query][:, columns] - mean) / scale,
                                    np.ones(len(query))))
    use_support = support_weight > 0
    classes = np.unique(np.r_[labels, target[support]] if use_support else labels)
    y = np.searchsorted(classes, labels)
    day_one_counts = np.bincount(y, minlength=len(classes))
    per_class_mass = len(labels) / len(np.unique(labels))
    day_one_weights = per_class_mass / day_one_counts[y]
    if use_support:
        ys = np.searchsorted(classes, target[support])
        support_counts = np.bincount(ys, minlength=len(classes))
        support_weights = support_weight * per_class_mass / support_counts[ys]
        design = np.vstack((baseline, support_design))
        ids = np.r_[y, ys]
        weights = np.r_[day_one_weights, support_weights]
    else:
        design, ids, weights = baseline, y, day_one_weights
    regularizer = np.eye(design.shape[1]) * alpha
    regularizer[-1, -1] = 0
    coefficients = np.linalg.solve(design.T @ (design * weights[:, None]) + regularizer,
                                   design.T @ (np.eye(len(classes))[ids] * weights[:, None]))
    predictions = classes[(query_design @ coefficients).argmax(axis=1)]
    return float(balanced_accuracy_score(target[query], predictions))
