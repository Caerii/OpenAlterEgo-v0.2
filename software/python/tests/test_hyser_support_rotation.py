import numpy as np
import pytest

from openalterego.research.hyser_fewshot import support_query
from scripts.hyser_support_rotation_stress import support_at_rank


def test_support_rotation_is_disjoint_and_rank_zero_matches_reference():
    labels = np.asarray([0, 0, 1, 1, 0, 1])
    first, first_query = support_at_rank(labels, 0)
    assert np.array_equal(first, support_query(labels, 1)[0])
    assert np.array_equal(first_query, support_query(labels, 1)[1])
    later, later_query = support_at_rank(labels, 1)
    assert set(later).isdisjoint(later_query)
    assert set(later) | set(later_query) == set(range(len(labels)))
    assert all(labels[later] == [0, 1])
    with pytest.raises(ValueError):
        support_at_rank(labels, 3)
