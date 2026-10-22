import numpy as np

from scripts.gaddy_channel_search import channel_local_lags, normalize_rows


def test_lag_design_keeps_eight_features_per_channel_and_uses_past_only():
    features = np.arange(6 * 16, dtype=np.float64).reshape(6, 16)
    design = channel_local_lags(features)
    assert design.shape == (6, 64)
    np.testing.assert_array_equal(design[0, :8],
                                  np.tile(features[0, [0, 8]], 4))
    np.testing.assert_array_equal(design[5, :8],
                                  features[[5, 5, 4, 4, 3, 3, 1, 1],
                                           [0, 8] * 4])
    np.testing.assert_array_equal(design[5, 8:16],
                                  features[[5, 5, 4, 4, 3, 3, 1, 1],
                                           [1, 9] * 4])


def test_session_normalizer_uses_only_calibration_features():
    calibration = [(np.full((3, 16), 2.0), np.zeros(3))]
    later = [(np.full((2, 16), 5.0), np.ones(2))]
    normalized = normalize_rows(later, calibration)
    np.testing.assert_allclose(normalized[0][0], 30.0)
    np.testing.assert_array_equal(normalized[0][1], np.ones(2))
