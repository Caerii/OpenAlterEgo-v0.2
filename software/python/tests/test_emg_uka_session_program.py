import numpy as np
import pytest

from scripts.emg_uka_session_program import (classify, load_adc,
                                             utterance_features)


def test_adc_crop_ignores_marker_and_rejects_invalid_offsets(tmp_path):
    samples = np.arange(21 * 7, dtype="<i2").reshape(21, 7)
    samples[:, 6] = 32000
    adc = tmp_path / "recording.adc"
    offset = tmp_path / "offset.txt"
    adc.write_bytes(samples.tobytes())
    offset.write_text("0 0\n2 20\n")
    cropped = load_adc(adc, offset)
    np.testing.assert_array_equal(cropped, samples[2:20, :6])
    assert utterance_features(cropped).shape == (60,)
    offset.write_text("0 0\n2 200\n")
    with pytest.raises(ValueError, match="outside recording"):
        load_adc(adc, offset)


def test_ridge_predictions_depend_only_on_selected_recorded_channels():
    rng = np.random.default_rng(7)
    train = rng.normal(size=(30, 60))
    labels = np.tile(np.arange(10), 3)
    query = rng.normal(size=(10, 60))
    baseline = classify(train, labels, query, (0, 1, 2))
    train[:, 30:] *= 1000
    query[:, 30:] *= -1000
    np.testing.assert_array_equal(classify(train, labels, query,
                                           (0, 1, 2)), baseline)
