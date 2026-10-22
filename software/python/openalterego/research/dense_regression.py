"""Batched ridge regression for channel-local measurement programs."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RegressionFold:
    gram: np.ndarray
    cross: np.ndarray
    validation_design: np.ndarray
    validation_target: np.ndarray
    channel_count: int
    features_per_channel: int


def prepare_regression_fold(train: tuple[np.ndarray, np.ndarray],
                            validation: tuple[np.ndarray, np.ndarray], *,
                            features_per_channel: int) -> RegressionFold:
    x, y = (np.asarray(value, dtype=np.float64) for value in train)
    val_x, val_y = (np.asarray(value, dtype=np.float64)
                    for value in validation)
    if (x.ndim != 2 or val_x.ndim != 2 or y.ndim != 1 or val_y.ndim != 1 or
            features_per_channel < 1 or x.shape[1] < features_per_channel or
            x.shape[1] % features_per_channel or
            val_x.shape[1] != x.shape[1] or len(x) != len(y) or
            len(val_x) != len(val_y) or not len(x) or not len(val_x) or
            not all(np.isfinite(value).all()
                    for value in (x, y, val_x, val_y))):
        raise ValueError("invalid channel-local regression arrays")
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1e-6)
    design = np.column_stack(((x - mean) / scale, np.ones(len(x))))
    validation_design = np.column_stack(
        ((val_x - mean) / scale, np.ones(len(val_x))))
    return RegressionFold(design.T @ design, design.T @ y,
                          validation_design, val_y,
                          x.shape[1] // features_per_channel,
                          features_per_channel)


def predict_regression_programs(fold: RegressionFold,
                                programs: np.ndarray, *,
                                alpha: float = 100.0,
                                batch_size: int = 64) -> np.ndarray:
    """Return one validation prediction column per proposed channel program."""
    programs = np.asarray(programs, dtype=np.int16)
    if (programs.ndim != 2 or not len(programs) or
            not 1 <= programs.shape[1] <= fold.channel_count or
            np.any((programs < 0) | (programs >= fold.channel_count)) or
            any(len(set(row)) != len(row) for row in programs) or
            alpha <= 0 or batch_size < 1):
        raise ValueError("invalid regression program batch")
    predictions = np.empty((len(fold.validation_target), len(programs)))
    feature_count = fold.channel_count * fold.features_per_channel
    for start in range(0, len(programs), batch_size):
        block = programs[start:start + batch_size]
        columns = (block[:, :, None] * fold.features_per_channel +
                   np.arange(fold.features_per_channel)).reshape(len(block), -1)
        columns = np.column_stack((columns,
                                   np.full(len(block), feature_count)))
        regularizer = np.eye(columns.shape[1]) * alpha
        regularizer[-1, -1] = 0
        coefficients = np.linalg.solve(
            fold.gram[columns[:, :, None], columns[:, None, :]] + regularizer,
            fold.cross[columns, None])[..., 0]
        predictions[:, start:start + len(block)] = np.einsum(
            "nbd,bd->nb", fold.validation_design[:, columns], coefficients)
    return predictions


def regression_mae(fold: RegressionFold, programs: np.ndarray, *,
                   alpha: float = 100.0) -> np.ndarray:
    predictions = predict_regression_programs(fold, programs, alpha=alpha)
    return np.mean(np.abs(predictions - fold.validation_target[:, None]),
                   axis=0)
