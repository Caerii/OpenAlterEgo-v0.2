import numpy as np

from scripts.emg_uka_session_drift import estimate_shift


def test_channel_feature_shift_recovers_shared_bins_from_prefix():
    shift = np.arange(12, dtype=np.float64).reshape(6, 1, 2)
    calibration = np.broadcast_to(shift, (3, 6, 5, 2)).reshape(3, 60)
    np.testing.assert_allclose(estimate_shift(calibration,
                                              "channel_feature"),
                               calibration[0])
    np.testing.assert_allclose(estimate_shift(calibration, "none"), 0)
    assert not np.allclose(estimate_shift(calibration, "global"),
                           calibration[0])
