import numpy as np
import pytest

from openalterego.research.guided_program_search import guided_search


def test_guided_search_scores_unique_complete_programs_at_exact_budget():
    options = tuple(tuple((base + i,) for i in range(5))
                    for base in (0, 10, 20, 30))
    def score(programs):
        assert programs.shape[1] == 4
        return -np.square(programs - np.array([4, 14, 24, 34])).sum(axis=1)

    programs, scores, winners = guided_search(
        options, score, budget=100, batch_size=20, seed=7)
    repeat = guided_search(options, score, budget=100, batch_size=20, seed=7)
    assert programs.shape == (100, 4)
    assert len({tuple(row) for row in programs}) == 100
    assert set(winners) == {20, 40, 60, 80, 100}
    assert all(winners[k] < k for k in winners)
    assert np.array_equal(programs, repeat[0])
    assert np.array_equal(scores, repeat[1])
    assert winners == repeat[2]
    assert [scores[winners[k]] for k in winners] == sorted(
        scores[winners[k]] for k in winners)


def test_guided_search_rejects_exhausted_space():
    with pytest.raises(ValueError, match="unable to sample"):
        guided_search((((0,), (1,)),), lambda x: np.zeros(len(x)),
                      budget=3, batch_size=1)
