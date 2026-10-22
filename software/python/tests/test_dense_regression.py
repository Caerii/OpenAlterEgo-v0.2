import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from openalterego.research.dense_regression import (
    predict_regression_programs, prepare_regression_fold, regression_mae)


def test_batched_program_predictions_match_independent_ridge_fits():
    rng = np.random.default_rng(82)
    train_x = rng.normal(size=(90, 24))
    train_y = train_x[:, 0] - 0.4 * train_x[:, 17] + rng.normal(
        scale=0.2, size=90)
    val_x = rng.normal(size=(25, 24))
    val_y = val_x[:, 0] - 0.4 * val_x[:, 17]
    fold = prepare_regression_fold((train_x, train_y), (val_x, val_y),
                                   features_per_channel=4)
    programs = np.asarray([(0, 4), (1, 3), (0, 2)],
                          dtype=np.int16)
    predictions = predict_regression_programs(fold, programs, alpha=10)
    for index, program in enumerate(programs):
        columns = [channel * 4 + offset for channel in program
                   for offset in range(4)]
        reference = make_pipeline(StandardScaler(), Ridge(alpha=10)).fit(
            train_x[:, columns], train_y).predict(val_x[:, columns])
        np.testing.assert_allclose(predictions[:, index], reference,
                                   rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(regression_mae(fold, programs, alpha=10),
                               np.abs(predictions - val_y[:, None]).mean(axis=0))


def test_invalid_programs_and_arrays_fail_closed():
    x = np.ones((5, 4))
    y = np.arange(5)
    with pytest.raises(ValueError):
        prepare_regression_fold((x, y), (x[:, :3], y),
                                features_per_channel=2)
    fold = prepare_regression_fold((x, y), (x, y),
                                   features_per_channel=2)
    with pytest.raises(ValueError):
        regression_mae(fold, np.asarray([[0, 0]]))
