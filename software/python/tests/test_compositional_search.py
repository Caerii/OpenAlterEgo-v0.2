import numpy as np
import pytest

from openalterego.research.compositional_search import beam_search, random_search


def test_batched_beam_and_random_search_cover_small_space():
    options = (((0, 1), (2, 3)), ((4, 5), (6, 7)))

    def score(programs):
        return (programs == 2).sum(axis=1) + (programs == 6).sum(axis=1)

    beam = beam_search(options, score, beam_size=2)
    random = random_search(options, score, evaluations=4, seed=2)
    assert beam.channels == random.channels == (2, 3, 6, 7)
    assert beam.stage_evaluations == (2, 4)
    assert beam.evaluations == 6
    assert random.evaluations == 4
    with pytest.raises(ValueError):
        beam_search((((0, 0),),), score)


def test_partial_scores_can_miss_an_interaction():
    options = (((0,), (1,)), ((2,), (3,)))

    def score(programs):
        return np.where(programs.shape[1] == 1,
                        (programs == 0).sum(axis=1),
                        10 * ((programs == 1).any(axis=1) &
                              (programs == 3).any(axis=1)))

    assert beam_search(options, score, beam_size=1).channels == (0, 2)
    assert random_search(options, score, evaluations=4).channels == (1, 3)
