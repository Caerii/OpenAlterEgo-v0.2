import numpy as np
import pytest

from scripts.hyser_differential_budget_audit import budget_curve


def test_nested_budget_curve_can_expose_winners_curse():
    cv = np.arange(10, dtype=float)
    future = -cv
    result = budget_curve(cv, future, (1, 3, 10), repetitions=10, seed=5)
    assert result["mean_day1_cv"][0] <= result["mean_day1_cv"][1]
    assert result["mean_day1_cv"][1] <= result["mean_day1_cv"][2]
    assert result["mean_selected_day2"][0] >= result["mean_selected_day2"][1]
    assert result["mean_selected_day2"][1] >= result["mean_selected_day2"][2]
    assert result == budget_curve(cv, future, (1, 3, 10), repetitions=10, seed=5)
    with pytest.raises(ValueError):
        budget_curve(cv, future, (11,), repetitions=1, seed=5)
