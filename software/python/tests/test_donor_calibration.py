import numpy as np
import pytest

from openalterego.research.donor_calibration import predict_left_out_person


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
