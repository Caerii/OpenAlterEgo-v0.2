import numpy as np
import pytest

from scripts.gaddy_prosody_probe import (causal_filter_emg, estimate_f0, normalize_from_calibration,
                                         official_offline_emg, robust_session_normalize,
                                         paired_bootstrap_interval, score, trial_frames)


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


def test_official_offline_preprocessing_rejects_mains_and_preserves_frameability():
    t = np.arange(2000) / 1000
    x = np.sin(2 * np.pi * 60 * t) + 0.25 * np.sin(2 * np.pi * 120 * t)
    raw = np.tile(x[:, None], (1, 8))
    prepared = official_offline_emg(raw, raw, raw)
    assert prepared.shape[1] == 8
    assert len(prepared) == len(np.arange(0, (len(raw) - 1) / 1000, 1 / 516.79))
    assert np.sqrt(np.mean(prepared[100:-100] ** 2)) < 0.1
    features, _, _ = trial_frames(prepared, np.zeros(32000), 516.79)
    assert len(features) > 0


def test_causal_filter_does_not_change_past_when_future_changes():
    rng = np.random.default_rng(17)
    raw = rng.normal(size=(2000, 8))
    changed = raw.copy()
    changed[1000:] *= 100
    assert np.array_equal(causal_filter_emg(raw)[:1000], causal_filter_emg(changed)[:1000])


def test_utterance_bootstrap_keeps_paired_gain_units():
    lower, upper = paired_bootstrap_interval(np.array([0.1, 0.2, 0.3, 0.4]))
    assert 0 < lower <= 0.25 <= upper
    assert paired_bootstrap_interval(np.array([-0.2] * 4)) == pytest.approx([-0.2, -0.2])
