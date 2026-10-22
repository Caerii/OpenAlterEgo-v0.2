"""Batched ridge screening over dense, recorded EMG electrode subsets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold


@dataclass
class Fold:
    gram: np.ndarray
    cross: np.ndarray
    validation_design: np.ndarray
    validation_labels: np.ndarray
    classes: np.ndarray


def prepare_folds(features: np.ndarray, labels: np.ndarray, *, seed: int = 42,
                  folds: int = 3) -> list[Fold]:
    if (features.ndim != 2 or features.shape[1] != 512 or
            labels.shape != (len(features),) or
            not np.isfinite(features).all() or
            len(np.unique(labels)) < 30 or
            not set(np.unique(labels)).issubset(set(range(1, 35)))):
        raise ValueError("expected at least 30 gesture classes and 256 channels")
    classes = np.unique(labels)
    n_classes = len(classes)
    result = []
    for train_ids, val_ids in StratifiedKFold(
            n_splits=folds, shuffle=True, random_state=seed).split(features, labels):
        train_x = features[train_ids].astype(np.float64)
        val_x = features[val_ids].astype(np.float64)
        y = np.searchsorted(classes, labels[train_ids])
        mean = train_x.mean(axis=0)
        scale = np.maximum(train_x.std(axis=0), 1e-6)
        design = np.column_stack(((train_x - mean) / scale, np.ones(len(train_x))))
        validation = np.column_stack(((val_x - mean) / scale, np.ones(len(val_x))))
        weights = len(y) / (n_classes * np.bincount(y, minlength=n_classes)[y])
        onehot = np.eye(n_classes)[y]
        result.append(Fold(design.T @ (design * weights[:, None]),
                           design.T @ (onehot * weights[:, None]),
                           validation, np.searchsorted(classes, labels[val_ids]),
                           classes))
    return result


def score_designs(folds: list[Fold], designs: np.ndarray, *, alpha: float = 10.0,
                  batch_size: int = 64) -> np.ndarray:
    """Score many channel programs from shared fold sufficient statistics."""
    designs = np.asarray(designs, dtype=np.int16)
    if (designs.ndim != 2 or not 1 <= designs.shape[1] <= 256 or
            np.any((designs < 0) | (designs >= 256)) or
            any(len(set(row)) != len(row) for row in designs) or
            not folds or alpha <= 0 or batch_size < 1):
        raise ValueError("invalid recorded-channel designs or screening parameters")
    result = np.zeros(len(designs), dtype=np.float64)
    for start in range(0, len(designs), batch_size):
        block = designs[start:start + batch_size]
        columns = np.column_stack((2 * block, 2 * block + 1)).reshape(len(block), -1)
        columns = np.column_stack((columns, np.full(len(block), 512)))
        dimension = columns.shape[1]
        regularizer = np.eye(dimension) * alpha
        regularizer[-1, -1] = 0
        for fold in folds:
            local_gram = fold.gram[columns[:, :, None], columns[:, None, :]]
            coefficients = np.linalg.solve(local_gram + regularizer,
                                           fold.cross[columns])
            predictions = np.einsum("nbd,bde->nbe",
                                    fold.validation_design[:, columns],
                                    coefficients).argmax(axis=2)
            recalls = np.stack([(predictions[fold.validation_labels == label] == label).mean(axis=0)
                                for label in range(len(fold.classes))])
            result[start:start + len(block)] += recalls.mean(axis=0) / len(folds)
    return result


def feasible(design: tuple[int, ...], sites: np.ndarray, *, min_grid_distance: int = 2) -> bool:
    """Enforce a topological separation within each of four recorded grids."""
    if any(index < 0 or index >= len(sites) for index in design) or len(set(design)) != len(design):
        return False
    for left, index in enumerate(design):
        for other in design[left + 1:]:
            if sites[index, 0] == sites[other, 0] and np.sum(
                    (sites[index, 1:] - sites[other, 1:]) ** 2) < min_grid_distance ** 2:
                return False
    return True


def greedy_search(folds: list[Fold], sites: np.ndarray, *, width: int = 8) -> tuple[list[int], float, int]:
    chosen = []
    evaluated = 0
    for _ in range(width):
        candidates = [index for index in range(256)
                      if feasible(tuple(chosen + [index]), sites)]
        programs = np.asarray([chosen + [index] for index in candidates], dtype=np.int16)
        scores = score_designs(folds, programs)
        evaluated += len(candidates)
        best = min(range(len(candidates)), key=lambda i: (-scores[i], candidates[i]))
        chosen.append(candidates[best])
    return chosen, float(scores[best]), evaluated


def random_budget_search(folds: list[Fold], sites: np.ndarray, *, width: int,
                         budget: int, seed: int = 42) -> tuple[list[int], float]:
    rng = np.random.default_rng(seed)
    programs = []
    seen = set()
    while len(programs) < budget:
        design = tuple(sorted(map(int, rng.choice(256, size=width, replace=False))))
        if design not in seen and feasible(design, sites):
            programs.append(design)
            seen.add(design)
    scores = score_designs(folds, np.asarray(programs, dtype=np.int16))
    best = int(np.argmax(scores))
    return list(programs[best]), float(scores[best])


def ridge_test(train: tuple[np.ndarray, np.ndarray], test: tuple[np.ndarray, np.ndarray],
               channels: list[int], *, alpha: float = 10.0) -> float:
    x, labels = train
    xt, target = test
    if x.shape[1] != 512 or xt.shape[1] != 512:
        raise ValueError("expected 256 channels with two features each")
    columns = [column for channel in channels for column in (2 * channel, 2 * channel + 1)]
    mean = x[:, columns].mean(axis=0)
    scale = np.maximum(x[:, columns].std(axis=0), 1e-6)
    design = np.column_stack(((x[:, columns] - mean) / scale, np.ones(len(x))))
    validation = np.column_stack(((xt[:, columns] - mean) / scale, np.ones(len(xt))))
    classes = np.unique(labels)
    y = np.searchsorted(classes, labels)
    weights = len(y) / (len(classes) * np.bincount(y, minlength=len(classes))[y])
    regularizer = np.eye(design.shape[1]) * alpha
    regularizer[-1, -1] = 0
    coefficients = np.linalg.solve(design.T @ (design * weights[:, None]) + regularizer,
                                   design.T @ (np.eye(len(classes))[y] * weights[:, None]))
    return float(balanced_accuracy_score(target,
                                         classes[(validation @ coefficients).argmax(axis=1)]))
