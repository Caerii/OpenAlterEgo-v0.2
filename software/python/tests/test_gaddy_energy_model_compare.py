import numpy as np

from scripts.gaddy_energy_model_compare import interior_voiced, lag_features, shift_past


def test_lag_and_misalignment_use_only_same_utterance_past():
    second = np.array([[100.0], [101.0], [102.0]])
    assert np.array_equal(lag_features(second, (0, 1, 2)),
                          np.array([[100., 100., 100.], [101., 100., 100.], [102., 101., 100.]]))
    assert np.array_equal(shift_past(second, 2).ravel(), [100., 100., 100.])


def test_voiced_interior_excludes_boundary_and_unvoiced_frames():
    pitch = np.array([0., 200., 200., 200., 200., 200., 0., 200.])
    assert np.array_equal(interior_voiced(pitch),
                          [False, False, False, True, False, False, False, False])
