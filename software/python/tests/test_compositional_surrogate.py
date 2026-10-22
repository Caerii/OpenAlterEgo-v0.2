import numpy as np
import pytest

from openalterego.research.compositional_surrogate import (
    fit_additive_program_model, fit_binary_program_model)


def test_additive_model_ranks_unseen_combinations():
    rng = np.random.default_rng(11)
    weights = rng.normal(size=(3, 5))
    train = rng.integers(5, size=(1000, 3), dtype=np.int16)
    target = weights[np.arange(3), train].sum(axis=1)
    model = fit_additive_program_model(train, target, (5, 5, 5), ridge=.01)
    unseen = rng.integers(5, size=(100, 3), dtype=np.int16)
    actual = weights[np.arange(3), unseen].sum(axis=1)
    predicted = model.predict(unseen)
    assert np.corrcoef(actual, predicted)[0, 1] > .999


def test_additive_model_rejects_invalid_codes():
    codes = np.asarray([[0, 0], [1, 1]], dtype=np.int16)
    model = fit_additive_program_model(codes, np.array([0., 1.]), (2, 2))
    with pytest.raises(ValueError, match="grammar codes"):
        model.predict(np.asarray([[2, 0]], dtype=np.int16))


def test_binary_program_model_ranks_unseen_subsets():
    rng = np.random.default_rng(17)
    features = np.zeros((1000, 9), dtype=np.float64)
    for row in features:
        row[rng.choice(8, size=4, replace=False)] = 1
    features[:, 8] = rng.integers(2, size=len(features))
    weights = rng.normal(size=9)
    scores = features @ weights
    model = fit_binary_program_model(features[:800], scores[:800], ridge=.01)
    assert np.corrcoef(model.predict(features[800:]), scores[800:])[0, 1] > .999
