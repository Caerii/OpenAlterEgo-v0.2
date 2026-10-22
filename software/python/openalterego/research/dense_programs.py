"""Batched ridge evaluation of channel programs with arbitrary local features."""

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedKFold


@dataclass
class DenseFold:
    gram: np.ndarray
    cross: np.ndarray
    validation_design: np.ndarray
    validation_labels: np.ndarray
    train_classes: np.ndarray


def prepare_fold(train: tuple[np.ndarray, np.ndarray],
                 validation: tuple[np.ndarray, np.ndarray],
                 *, features_per_site: int) -> DenseFold:
    x, labels = train
    xt, target = validation
    width = x.shape[1] if x.ndim == 2 else 0
    if (features_per_site < 1 or x.ndim != 2 or xt.ndim != 2 or
            width < features_per_site or width % features_per_site or
            xt.shape[1] != width or
            len(labels) != len(x) or len(target) != len(xt) or
            not np.isfinite(x).all() or not np.isfinite(xt).all()):
        raise ValueError("invalid channel-local feature matrices")
    classes, y = np.unique(labels, return_inverse=True)
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1e-6)
    design = np.column_stack(((x - mean) / scale, np.ones(len(x))))
    validation_design = np.column_stack(((xt - mean) / scale, np.ones(len(xt))))
    weights = len(y) / (len(classes) * np.bincount(y)[y])
    onehot = np.eye(len(classes))[y]
    return DenseFold(design.T @ (design * weights[:, None]),
                     design.T @ (onehot * weights[:, None]),
                     validation_design, np.asarray(target), classes)


def prepare_cv_folds(features: np.ndarray, labels: np.ndarray,
                     *, features_per_site: int, seed: int = 42,
                     folds: int = 3) -> list[DenseFold]:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return [prepare_fold((features[train], labels[train]),
                         (features[val], labels[val]),
                         features_per_site=features_per_site)
            for train, val in splitter.split(features, labels)]


def score_programs(folds: list[DenseFold], designs: np.ndarray,
                   *, features_per_site: int, alpha: float = 10.0,
                   batch_size: int = 32) -> np.ndarray:
    """Mean balanced accuracy across validation folds."""
    return score_programs_by_fold(
        folds, designs, features_per_site=features_per_site,
        alpha=alpha, batch_size=batch_size).mean(axis=0)


def score_programs_by_fold(folds: list[DenseFold], designs: np.ndarray,
                           *, features_per_site: int, alpha: float = 10.0,
                           batch_size: int = 32) -> np.ndarray:
    """Balanced accuracy per fold and program, retaining temporal variation."""
    designs = np.asarray(designs, dtype=np.int16)
    channel_count = ((folds[0].gram.shape[0] - 1) // features_per_site
                     if folds and features_per_site >= 1 else 0)
    if (not folds or channel_count < 1 or designs.ndim != 2 or
            not 1 <= designs.shape[1] <= channel_count or
            np.any((designs < 0) | (designs >= channel_count)) or
            any(len(set(row)) != len(row) for row in designs) or
            features_per_site < 1 or alpha <= 0 or batch_size < 1):
        raise ValueError("invalid design batch")
    result = np.empty((len(folds), len(designs)), dtype=np.float64)
    bias = channel_count * features_per_site
    for start in range(0, len(designs), batch_size):
        block = designs[start:start + batch_size]
        columns = (block[:, :, None] * features_per_site +
                   np.arange(features_per_site)).reshape(len(block), -1)
        columns = np.column_stack((columns, np.full(len(block), bias)))
        regularizer = np.eye(columns.shape[1]) * alpha
        regularizer[-1, -1] = 0
        for fold_index, fold in enumerate(folds):
            coefficients = np.linalg.solve(
                fold.gram[columns[:, :, None], columns[:, None, :]] + regularizer,
                fold.cross[columns])
            predictions = fold.train_classes[np.einsum(
                "nbd,bde->nbe", fold.validation_design[:, columns],
                coefficients).argmax(axis=2)]
            recalls = [np.mean(predictions[fold.validation_labels == label] == label,
                               axis=0)
                       for label in np.unique(fold.validation_labels)]
            result[fold_index, start:start + len(block)] = np.mean(
                recalls, axis=0)
    return result
