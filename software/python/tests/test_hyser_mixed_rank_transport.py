import numpy as np

from scripts.hyser_mixed_rank_transport import split_remaining_repetitions, top_overlap


def test_query_halves_cover_each_class_without_overlap():
    labels = np.repeat(np.arange(3), 6)
    query = np.array([2, 3, 4, 5, 8, 9, 10, 11, 14, 15, 16, 17])
    first, second, excluded = split_remaining_repetitions(labels, query)
    assert set(first) | set(second) == set(query)
    assert not set(first) & set(second)
    assert np.array_equal(np.bincount(labels[first]), [2, 2, 2])
    assert np.array_equal(np.bincount(labels[second]), [2, 2, 2])
    assert len(excluded) == 0


def test_singleton_query_class_is_recorded_outside_both_halves():
    labels = np.array([0, 0, 0, 1, 1, 1])
    first, second, excluded = split_remaining_repetitions(
        labels, np.array([1, 2, 5]))
    assert first.tolist() == [1]
    assert second.tolist() == [2]
    assert excluded.tolist() == [5]


def test_top_overlap_is_fraction_of_shared_top_candidates():
    assert top_overlap(np.arange(10), np.arange(10)) == 1
    assert top_overlap(np.arange(10), -np.arange(10)) == 0
