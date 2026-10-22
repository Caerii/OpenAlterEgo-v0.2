import numpy as np
import pytest

from openalterego.research.hyser_pair_stability import (
    pair_drift_ranks, rank_programs)


def test_drift_rank_prefers_a_stable_informative_pair():
    rng = np.random.default_rng(13)
    labels = np.repeat([1, 2, 3], 12)
    base = rng.normal(size=(len(labels), 4)) * 0.1
    base[:, :2] += labels[:, None]
    later = base.copy()
    later[:, 2:] += 3.0
    sites = ["AA-1-1", "AA-1-2", "AA-1-3", "AA-1-4"] + [
        f"BB-{index // 8 + 1}-{index % 8 + 1}" for index in range(252)]
    pairs = [[0, 1], [2, 3]]
    rank = pair_drift_ranks((base, labels), (later, labels), pairs, sites)
    assert rank[0] < rank[1]
    scores = rank_programs(np.array([0.5, 0.5]),
                           np.array([[0], [1]]), rank, 0.1)
    assert scores[0] > scores[1]
    with pytest.raises(ValueError):
        rank_programs(np.array([0.5]), np.array([[2]]), rank, 0.1)
