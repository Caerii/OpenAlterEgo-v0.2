"""Batched next-day outcomes for many recorded-site design programs."""

from dataclasses import dataclass

import numpy as np


@dataclass
class DayPair:
    gram: np.ndarray
    cross: np.ndarray
    target_design: np.ndarray
    target_labels: np.ndarray
    train_classes: np.ndarray


def prepare_day_pair(train: tuple[np.ndarray, np.ndarray],
                     target: tuple[np.ndarray, np.ndarray]) -> DayPair:
    x, labels = train
    xt, target_labels = target
    if x.ndim != 2 or xt.ndim != 2 or x.shape[1] != 512 or xt.shape[1] != 512:
        raise ValueError("expected trial-by-512 feature matrices")
    if len(x) != len(labels) or len(xt) != len(target_labels):
        raise ValueError("features and labels differ in length")
    classes, y = np.unique(labels, return_inverse=True)
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1e-6)
    design = np.column_stack(((x - mean) / scale, np.ones(len(x))))
    target_design = np.column_stack(((xt - mean) / scale, np.ones(len(xt))))
    weights = len(y) / (len(classes) * np.bincount(y)[y])
    onehot = np.eye(len(classes))[y]
    return DayPair(design.T @ (design * weights[:, None]),
                   design.T @ (onehot * weights[:, None]),
                   target_design, target_labels, classes)


def score_day_pair(pair: DayPair, designs: np.ndarray, *, alpha: float = 10.0,
                   batch_size: int = 64) -> np.ndarray:
    """Balanced accuracy across all target classes, including unseen ones."""
    designs = np.asarray(designs, dtype=np.int16)
    if (designs.ndim != 2 or not 1 <= designs.shape[1] <= 256 or
            np.any((designs < 0) | (designs >= 256)) or
            any(len(set(row)) != len(row) for row in designs) or
            alpha <= 0 or batch_size < 1):
        raise ValueError("invalid design batch")
    result = np.empty(len(designs), dtype=np.float64)
    target_classes = np.unique(pair.target_labels)
    for start in range(0, len(designs), batch_size):
        block = designs[start:start + batch_size]
        columns = np.column_stack((2 * block, 2 * block + 1)).reshape(len(block), -1)
        columns = np.column_stack((columns, np.full(len(block), 512)))
        regularizer = np.eye(columns.shape[1]) * alpha
        regularizer[-1, -1] = 0
        gram = pair.gram[columns[:, :, None], columns[:, None, :]]
        coefficients = np.linalg.solve(gram + regularizer, pair.cross[columns])
        prediction_ids = np.einsum("nbd,bde->nbe",
                                   pair.target_design[:, columns],
                                   coefficients).argmax(axis=2)
        predictions = pair.train_classes[prediction_ids]
        recalls = [np.mean(predictions[pair.target_labels == label] == label, axis=0)
                   for label in target_classes]
        result[start:start + len(block)] = np.mean(recalls, axis=0)
    return result
