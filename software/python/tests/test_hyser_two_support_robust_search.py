import numpy as np

from scripts.hyser_two_support_robust_search import (
    BUDGETS, nested_budget_indices, two_support_split)


def test_two_support_rotations_leave_identical_disjoint_query():
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    first, second, support, query = two_support_split(labels)
    np.testing.assert_array_equal(first, [0, 1])
    np.testing.assert_array_equal(second, [2, 3])
    np.testing.assert_array_equal(support, [0, 1, 2, 3])
    np.testing.assert_array_equal(query, [4, 5, 6, 7])


def test_nested_budget_winner_converges_at_full_pool():
    scores = np.arange(max(BUDGETS), dtype=np.float64)
    choices = nested_budget_indices(scores, draws=3)
    assert choices[max(BUDGETS)] == [len(scores) - 1] * 3
    for draw in range(3):
        selected_scores = [scores[choices[budget][draw]]
                           for budget in BUDGETS]
        assert selected_scores == sorted(selected_scores)
