import numpy as np

from scripts.gaddy_contour_activity_probe import (make_model, paired_contour_advantage, past_smooth,
                                                   slow_contour, top_k_codes)


def test_past_smoothing_never_changes_earlier_frames_when_future_changes():
    original = np.arange(12, dtype=float)
    altered = original.copy()
    altered[6:] += 100
    assert np.array_equal(past_smooth(original)[:6], past_smooth(altered)[:6])
    assert past_smooth(original)[0] == original[0]


def test_slow_contour_excludes_cross_silence_differences():
    target = np.array([-3., -3., -1., -1., -1., -1., -1., -1., -3., -1., -1., -1., -1., -1.])
    row = (np.zeros((len(target), 16)), target, np.zeros(len(target)))
    result = slow_contour([row], [past_smooth(target)])
    assert result["frames"] == 3
    assert result["delta_200ms_mae"] == 0


def test_top_k_codes_retain_only_largest_components():
    codes = np.array([[1., 3., 2., 0.5], [4., 0., 0.1, 5.]])
    assert np.array_equal(top_k_codes(codes), [[0., 3., 2., 0.], [4., 0., 0., 5.]])


def test_paired_contour_advantage_uses_whole_utterances():
    target = np.linspace(-1, 1, 12)
    rows = [(np.zeros((12, 16)), target, np.zeros(12)) for _ in range(3)]
    chosen = [past_smooth(target) for _ in rows]
    control = [np.zeros_like(target) for _ in rows]
    result = paired_contour_advantage(rows, chosen, control)
    assert result["utterances"] == 3
    assert result["ci95"][0] > 0


def test_fast_fit_preset_changes_only_histogram_resolution():
    reference = make_model(True).get_params()
    fast = make_model(True, max_bins=64).get_params()
    assert reference["max_bins"] == 255
    assert fast["max_bins"] == 64
    assert {k: v for k, v in reference.items() if k != "max_bins"} == {
        k: v for k, v in fast.items() if k != "max_bins"}
