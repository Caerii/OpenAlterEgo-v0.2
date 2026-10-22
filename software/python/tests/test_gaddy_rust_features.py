import numpy as np
import pytest

from scripts.gaddy_prosody_probe import trial_frames


def test_rust_energy_features_match_python_when_extension_is_built():
    accel = pytest.importorskip("openalterego_accel")
    rng = np.random.default_rng(31)
    emg = rng.normal(size=(3237, 8))
    audio = rng.normal(size=52000)
    duration = min(len(emg) / 1000, len(audio) / 16000)
    centers = np.arange(0.15, duration - 0.15, 0.05)
    expected_x, expected_y, _ = trial_frames(emg, audio, estimate_pitch=False)
    actual_x, actual_y = accel.extract_energy_features(emg, audio, 1000.0, centers)
    np.testing.assert_allclose(actual_x, expected_x, rtol=0, atol=1e-12)
    np.testing.assert_allclose(actual_y, expected_y, rtol=0, atol=1e-12)


def test_existing_rust_spike_scatter_still_operates():
    accel = pytest.importorskip("openalterego_accel")
    output = np.zeros((8, 2), dtype=np.float32)
    accel.scatter_unit_multichannel(
        output, np.array([2], dtype=np.int32), 2.0,
        np.ones(8, dtype=np.float32), np.array([1., 0.5], dtype=np.float32),
        np.array([1., 3.], dtype=np.float32), 8, 2, 2,
    )
    np.testing.assert_array_equal(output[2:4], [[2., 6.], [1., 3.]])
