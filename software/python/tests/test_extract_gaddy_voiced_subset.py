import pytest

from scripts.extract_gaddy_voiced_subset import requested_components


def test_requested_range_is_inclusive_and_bounded():
    wanted = requested_components(100, 100)
    assert len(wanted) == 300
    assert "100_emg.npy" in wanted
    assert "199_audio_clean.flac" in wanted
    assert "99_emg.npy" not in wanted
    assert "200_emg.npy" not in wanted


@pytest.mark.parametrize("start,count", [(-1, 1), (0, 0), (0, 101)])
def test_invalid_ranges_rejected(start, count):
    with pytest.raises(ValueError):
        requested_components(start, count)
