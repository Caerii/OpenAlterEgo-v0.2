import numpy as np
import pytest

from openalterego.research.donor_calibration import nested_policy_gate, predict_left_out_person


def test_left_out_target_outcomes_cannot_change_prediction():
    rng = np.random.default_rng(29)
    proxies = rng.normal(size=(4, 15, 2))
    outcomes = rng.normal(size=(4, 15))
    expected, meta = predict_left_out_person(proxies, outcomes, 2)
    outcomes[2] = np.nan
    actual, repeated = predict_left_out_person(proxies, outcomes, 2)
    np.testing.assert_array_equal(actual, expected)
    assert meta["donor_indices"] == [0, 1, 3]
    np.testing.assert_array_equal(repeated["donor_prior"], meta["donor_prior"])


def test_nonfinite_donor_outcomes_are_rejected():
    proxies = np.ones((3, 4, 2))
    outcomes = np.ones((3, 4))
    outcomes[1, 0] = np.nan
    with pytest.raises(ValueError, match="donor outcomes"):
        predict_left_out_person(proxies, outcomes, 0)


def test_nested_gate_cannot_read_outer_target_query():
    rng = np.random.default_rng(13)
    proxies = rng.normal(size=(5, 12, 2))
    outcomes = rng.uniform(size=(5, 12))
    expected = nested_policy_gate(proxies, outcomes, 2, draws=200)
    outcomes[2] = np.nan
    assert nested_policy_gate(proxies, outcomes, 2, draws=200) == expected


def test_nested_gate_abstains_without_gain():
    proxies = np.ones((5, 4, 2))
    outcomes = np.ones((5, 4))
    gate = nested_policy_gate(proxies, outcomes, 0, draws=200)
    assert gate["selected_policy"] == "two_support"
    assert all(value["bootstrap_5th_percentile_gain"] == 0
               for value in gate["evidence"].values())


def test_nested_gate_accepts_consistent_donor_gain():
    proxies = np.zeros((5, 4, 2))
    proxies[:, 0, :] = 1
    outcomes = np.zeros((5, 4))
    outcomes[:, 1] = 1
    gate = nested_policy_gate(proxies, outcomes, 0, draws=200)
    assert gate["selected_policy"] in ("donor_only", "donor_calibrated")
    assert gate["evidence"][gate["selected_policy"]]["bootstrap_5th_percentile_gain"] > 0
