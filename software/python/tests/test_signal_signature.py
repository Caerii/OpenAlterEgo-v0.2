import numpy as np
import pytest

from openalterego.research.signal_signature import event_windows, signal_signature


def test_signature_detects_known_command_gain_and_spectral_shift():
    rng = np.random.default_rng(5)
    labels = np.repeat([0, 1, 0, 2], 500)
    signals = rng.normal(size=(len(labels), 14))
    signals[labels != 0] *= 3
    windows, classes = event_windows(signals, labels)
    report = signal_signature(windows, classes)
    assert report["windows"] == 4
    assert report["command_to_rest_rms_ratio"] == pytest.approx(3, rel=0.2)
    assert sum(report["band_power_fractions"].values()) == pytest.approx(1, abs=0.01)
    assert 0 <= report["median_absolute_channel_correlation"] <= 1


def test_signature_rejects_missing_rest_and_bad_shape():
    with pytest.raises(ValueError):
        signal_signature(np.ones((2, 400, 14)), np.ones(2))
    with pytest.raises(ValueError):
        event_windows(np.ones((10, 14)), np.ones(9))
