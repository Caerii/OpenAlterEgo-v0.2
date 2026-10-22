from pathlib import Path

import numpy as np

from openalterego.research.hyser_features import read_wfdb_16, site_coordinates, trial_features
from openalterego.research.hyser_search import feasible, prepare_folds, ridge_test, score_designs
from openalterego.research.hyser_site_health import dark_contact_mask
from openalterego.research.hyser_spatial_search import displacement_maps
from openalterego.research.hyser_transfer import prepare_day_pair, score_day_pair
from openalterego.research.hyser_temporal_features import temporal_trial_features
from openalterego.research.dense_programs import prepare_fold, score_programs
from openalterego.research.hyser_fewshot import ridge_query_score, support_query


def test_dense_screen_scores_batched_and_scalar_designs_identically():
    rng = np.random.default_rng(42)
    labels = np.repeat(np.arange(1, 35), 4)
    features = rng.normal(size=(len(labels), 512)).astype(np.float32)
    folds = prepare_folds(features, labels)
    designs = np.array([[0, 16, 64], [1, 80, 192]], dtype=np.int16)
    together = score_designs(folds, designs)
    separate = np.array([score_designs(folds, row[None, :])[0] for row in designs])
    np.testing.assert_allclose(together, separate, atol=1e-12)


def test_grid_spacing_uses_site_labels_not_flat_indices():
    sites = tuple(f"{grid}-{row}-{column}" for grid in ("ED", "EP", "FD", "FP")
                  for row in range(1, 9) for column in range(1, 9))
    coordinates = site_coordinates(sites)
    assert not feasible((0, 1), coordinates)
    assert feasible((0, 2, 64), coordinates)
    assert not feasible((0, 2, 4), coordinates, max_per_grid=2)
    assert feasible((0, 2, 64), coordinates, max_per_grid=2)


def test_dark_contact_screen_uses_neighbors_within_the_same_grid():
    sites = tuple(f"{grid}-{row}-{column}" for grid in ("ED", "EP", "FD", "FP")
                  for row in range(1, 9) for column in range(1, 9))
    coordinates = site_coordinates(sites)
    features = np.zeros((4, 512), dtype=np.float64)
    target = sites.index("ED-4-4")
    features[:, 2 * target] = -0.8
    allowed, darkness = dark_contact_mask(features, coordinates)
    assert not allowed[target]
    assert darkness[target] == 0.8
    assert allowed[sites.index("EP-4-4")]


def test_displacement_maps_follow_grid_labels():
    sites = tuple(f"{grid}-{row}-{column}" for grid in ("ED", "EP", "FD", "FP")
                  for row in range(1, 9) for column in range(1, 9))
    moves = displacement_maps(site_coordinates(sites))
    center = sites.index("ED-4-4")
    assert [sites[move[center]] for move in moves] == [
        "ED-3-4", "ED-5-4", "ED-4-3", "ED-4-5"]
    assert moves[0][sites.index("ED-1-4")] == -1


def test_batched_day_pair_matches_single_design_with_absent_train_class():
    rng = np.random.default_rng(9)
    labels = np.repeat(np.arange(1, 34), 5)
    target_labels = np.repeat(np.arange(1, 35), 2)
    train = (rng.normal(size=(len(labels), 512)), labels)
    target = (rng.normal(size=(len(target_labels), 512)), target_labels)
    designs = np.asarray([[0, 1, 64], [2, 70, 192]], dtype=np.int16)
    actual = score_day_pair(prepare_day_pair(train, target), designs)
    expected = [ridge_test(train, target, list(row)) for row in designs]
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_temporal_features_preserve_whole_trial_and_ordered_bins():
    signal = np.ones((8, 256), dtype=np.float32)
    signal[:2, 0] = 2
    signal[2:4, 0] = 3
    signal[4:6, 0] = 4
    signal[6:8, 0] = 5
    result = temporal_trial_features(signal).reshape(256, 10)
    np.testing.assert_allclose(result[:, :2], trial_features(signal).reshape(256, 2))
    np.testing.assert_allclose(result[0, [2, 4, 6, 8]], np.log([2, 3, 4, 5]))


def test_generic_dense_program_scorer_matches_two_feature_reference():
    rng = np.random.default_rng(19)
    labels = np.repeat(np.arange(1, 34), 5)
    target_labels = np.repeat(np.arange(1, 35), 2)
    train = (rng.normal(size=(len(labels), 512)), labels)
    target = (rng.normal(size=(len(target_labels), 512)), target_labels)
    designs = np.asarray([[0, 1, 64], [2, 70, 192]], dtype=np.int16)
    fold = prepare_fold(train, target, features_per_site=2)
    actual = score_programs([fold], designs, features_per_site=2)
    expected = [ridge_test(train, target, list(row)) for row in designs]
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_fewshot_support_is_disjoint_and_zero_weight_matches_reference():
    rng = np.random.default_rng(23)
    labels = np.repeat(np.arange(1, 34), 5)
    target_labels = np.repeat(np.arange(1, 35), 3)
    train = (rng.normal(size=(len(labels), 512)), labels)
    target = (rng.normal(size=(len(target_labels), 512)), target_labels)
    support, query = support_query(target_labels, 1)
    assert len(support) == 34 and len(query) == 68
    assert not set(support) & set(query)
    channels = [0, 1, 64]
    actual = ridge_query_score(train, target, channels, features_per_site=2,
                               alpha=10, support=support, query=query,
                               support_weight=0)
    expected = ridge_test(train, (target[0][query], target[1][query]), channels)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_wfdb_reader_checks_interleaving_and_header_checksum(tmp_path: Path):
    base = tmp_path / "dynamic_preprocess_sample1"
    sites = [f"{grid}-{row}-{column}" for grid in ("ED", "EP", "FD", "FP")
             for row in range(1, 9) for column in range(1, 9)]
    raw = np.tile(np.arange(256, dtype=np.int16), (4, 1))
    raw.tofile(base.with_suffix(".dat"))
    header = [f"{base.name} 256 2048 4"]
    header += [f"{base.name}.dat 16 2(0)/V 0 0 {index} {4 * index} 0 {site}"
               for index, site in enumerate(sites)]
    base.with_suffix(".hea").write_text("\n".join(header) + "\n", encoding="ascii")
    signal, names = read_wfdb_16(base)
    assert names == tuple(sites)
    assert signal.shape == (4, 256)
    assert signal[0, 2] == 1.0
    assert trial_features(signal).shape == (512,)
    header[1] = header[1].replace(" 0 0 ED-1-1", " 1 0 ED-1-1")
    base.with_suffix(".hea").write_text("\n".join(header) + "\n", encoding="ascii")
    import pytest
    with pytest.raises(ValueError, match="checksum"):
        read_wfdb_16(base)
