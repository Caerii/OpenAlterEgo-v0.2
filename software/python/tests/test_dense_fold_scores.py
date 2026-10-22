import numpy as np

from openalterego.research.dense_programs import (
    prepare_fold, score_programs, score_programs_by_fold)


def test_per_fold_scores_match_independent_scoring_and_existing_mean():
    rng = np.random.default_rng(420)
    train = (rng.normal(size=(60, 8)), np.tile(np.arange(3), 20))
    val_a = (rng.normal(size=(24, 8)), np.tile(np.arange(3), 8))
    val_b = (rng.normal(size=(30, 8)), np.tile(np.arange(3), 10))
    folds = [prepare_fold(train, val, features_per_site=2)
             for val in (val_a, val_b)]
    programs = np.asarray([[0, 1], [1, 3], [0, 2]], dtype=np.int16)
    individual = score_programs_by_fold(
        folds, programs, features_per_site=2, batch_size=2)
    assert individual.shape == (2, 3)
    for index, fold in enumerate(folds):
        np.testing.assert_allclose(individual[index], score_programs(
            [fold], programs, features_per_site=2, batch_size=2))
    np.testing.assert_allclose(individual.mean(axis=0), score_programs(
        folds, programs, features_per_site=2, batch_size=2))


def test_brier_score_matches_explicit_ridge_probabilities():
    rng = np.random.default_rng(37)
    train = (rng.normal(size=(30, 4)), np.tile(np.arange(3), 10))
    val = (rng.normal(size=(12, 4)), np.tile(np.arange(3), 4))
    fold = prepare_fold(train, val, features_per_site=2)
    program = np.array([[0, 1]], dtype=np.int16)
    observed = score_programs([fold], program, features_per_site=2,
                              metric="brier")[0]
    regularizer = np.eye(5) * 10.0
    regularizer[-1, -1] = 0
    coefficients = np.linalg.solve(fold.gram + regularizer, fold.cross)
    logits = fold.validation_design @ coefficients
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    trial_scores = (.5 + probabilities[np.arange(12), val[1]] -
                    .5 * np.square(probabilities).sum(axis=1))
    expected = np.mean([trial_scores[val[1] == label].mean()
                        for label in np.unique(val[1])])
    np.testing.assert_allclose(observed, expected)


def test_brier_score_handles_unseen_validation_class_as_zero_probability():
    train = (np.array([[0., 1.], [1., 0.], [0., 2.], [2., 0.]]),
             np.array([0, 1, 0, 1]))
    validation = (np.array([[1., 1.], [1., 2.]]), np.array([0, 2]))
    fold = prepare_fold(train, validation, features_per_site=1)
    value = score_programs([fold], np.array([[0, 1]]),
                           features_per_site=1, metric="brier")
    assert value.shape == (1,)
    assert np.isfinite(value).all()
    assert 0 <= value[0] <= 1
