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
