import pytest

from openalterego.research.rank_stability import batch_robust_choices, compare_rankings


def test_rank_stability_exposes_future_rank_reversal_and_regret():
    first = {("none", (0,)): 0.9, ("none", (1,)): 0.8,
             ("none", (2,)): 0.7}
    later = {("none", (0,)): 0.1, ("none", (1,)): 0.8,
             ("none", (2,)): 0.7}
    result = compare_rankings(first, later, top_sizes=(1, 2))
    assert result["top_overlap"] == {"1": 0, "2": 1}
    assert result["later_proxy_regret"] == pytest.approx(0.7)
    assert result["spearman_score_rank"] == pytest.approx(-0.5)


def test_rank_stability_rejects_mismatched_programs():
    with pytest.raises(ValueError, match="same programs"):
        compare_rankings({"a": 1, "b": 0}, {"a": 1, "c": 0})


def test_batch_penalty_can_reject_unstable_validation_winner():
    batches = [{("unstable",): 1.0, ("stable",): 0.6},
               {("unstable",): 0.4, ("stable",): 0.6}]
    future = {("unstable",): 0.2, ("stable",): 0.7}
    result = batch_robust_choices(batches, future, penalties=(0.0, 1.0))
    assert result["0.0"]["choice"] == ["unstable"]
    assert result["1.0"]["choice"] == ["stable"]
    assert result["1.0"]["later_proxy_regret"] == pytest.approx(0)
