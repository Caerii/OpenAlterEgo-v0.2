import numpy as np
import pytest

from scripts.gaddy_prosody_probe import (estimate_f0, normalize_from_calibration,
                                         robust_session_normalize, score, trial_frames)


def test_pitch_probe_finds_clean_sine_and_rejects_silence():
    time = np.arange(640) / 16000
    frequency, confidence = estimate_f0(np.sin(2 * np.pi * 200 * time))
    assert frequency == pytest.approx(200, abs=5)
    assert confidence > 0.45
    assert estimate_f0(np.zeros(640)) == (0.0, 0.0)


def test_trial_frames_and_split_score_are_finite():
    emg = np.ones((1000, 8))
    audio = np.sin(2 * np.pi * 200 * np.arange(16000) / 16000)
    features, energy, pitch = trial_frames(emg, audio)
    assert features.shape[1] == 16
    assert len(features) == len(energy) == len(pitch)
    assert np.isfinite(features).all()
    result = score(np.arange(40).reshape(20, 2), np.arange(20),
                   np.arange(20).reshape(10, 2), np.arange(10))
    assert np.isfinite(result["emg_mae"])


def test_session_normalization_removes_unlabeled_shift():
    base = np.arange(40, dtype=float).reshape(20, 2)
    assert np.allclose(robust_session_normalize(base),
                       robust_session_normalize(base * 3 + 10))


def test_calibration_only_normalization_uses_prior_frames():
    calibration = np.arange(40, dtype=float).reshape(20, 2)
    future = np.array([[1000.0, 2000.0]])
    expected = normalize_from_calibration(future, calibration)
    assert np.allclose(expected, normalize_from_calibration(future, calibration.copy()))
    assert not np.allclose(expected, robust_session_normalize(np.concatenate([calibration, future]))[-1])
