import numpy as np
import pytest

from openalterego.research.neighborhood_prior import one_swap_smoothed_scores


def test_one_swap_smoothing_preserves_constant_scores_and_normalization_boundary():
    from itertools import combinations

    subsets = list(combinations(range(4), 2))
    programs = [(norm, subset) for norm in ("none", "recording_robust")
                for subset in subsets]
    scores = np.r_[np.ones(6), np.full(6, 3.0)]
    np.testing.assert_allclose(one_swap_smoothed_scores(programs, scores,
                                                        n_channels=4, width=2), scores)


def test_one_swap_smoothing_damps_isolated_peak_and_rejects_missing_program():
    from itertools import combinations

    programs = [("none", subset) for subset in combinations(range(4), 2)]
    scores = np.zeros(6)
    scores[0] = 5
    smoothed = one_swap_smoothed_scores(programs, scores, n_channels=4, width=2)
    assert smoothed[0] == 1.0
    assert smoothed[-1] == 0.0  # Disjoint subset is two substitutions away.
    with pytest.raises(ValueError):
        one_swap_smoothed_scores(programs[:-1], scores[:-1], n_channels=4, width=2)
