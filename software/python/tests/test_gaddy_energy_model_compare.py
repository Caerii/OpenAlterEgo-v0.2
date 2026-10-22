from pathlib import Path

import numpy as np
import pytest

from scripts.gaddy_energy_model_compare import (feature_cache_path, interior_voiced,
                                                 lag_features, read_feature_cache,
                                                 shift_past, write_feature_cache)


def test_lag_and_misalignment_use_only_same_utterance_past():
    second = np.array([[100.0], [101.0], [102.0]])
    assert np.array_equal(lag_features(second, (0, 1, 2)),
                          np.array([[100., 100., 100.], [101., 100., 100.], [102., 101., 100.]]))
    assert np.array_equal(shift_past(second, 2).ravel(), [100., 100., 100.])


def test_voiced_interior_excludes_boundary_and_unvoiced_frames():
    pitch = np.array([0., 200., 200., 200., 200., 200., 0., 200.])
    assert np.array_equal(interior_voiced(pitch),
                          [False, False, False, True, False, False, False, False])


def test_feature_cache_preserves_utterance_boundaries_and_provenance(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    component = source / "0_info.json"
    component.write_text("first", encoding="utf-8")
    manifest = {"files": {"0_info.json": {"sha256": "illustrative"}}}
    cache = tmp_path / "cache"
    key_before = feature_cache_path(source, cache, manifest, "python", False)
    rows = [(np.ones((3, 16)), np.arange(3, dtype=float), np.zeros(3)),
            (np.full((2, 16), 2.), np.arange(2, dtype=float), np.zeros(2))]
    write_feature_cache(key_before, rows)
    loaded = read_feature_cache(key_before)
    assert [len(row[1]) for row in loaded] == [3, 2]
    for original, restored in zip(rows, loaded):
        for a, b in zip(original, restored):
            assert np.array_equal(a, b)
    component.write_text("changed source", encoding="utf-8")
    assert feature_cache_path(source, cache, manifest, "python", False) != key_before
    assert feature_cache_path(source, cache, manifest, "python", True) != key_before


def test_feature_cache_rejects_repository_location_and_bad_offsets(tmp_path):
    root = Path(__file__).resolve().parents[3]
    with pytest.raises(ValueError, match="outside the repository"):
        feature_cache_path(tmp_path, root / "feature-cache", {"files": {}}, "python", False)
    broken = tmp_path / "broken.npz"
    np.savez(broken, features=np.ones((2, 16)), energy=np.ones(2),
             pitch=np.zeros(2), offsets=np.array([0, 3]))
    with pytest.raises(ValueError, match="invalid feature cache"):
        read_feature_cache(broken)
