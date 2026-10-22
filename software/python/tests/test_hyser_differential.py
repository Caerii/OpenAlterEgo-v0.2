import numpy as np

from openalterego.research.hyser_differential import (
    adjacent_pairs, differential_trial_features)
from openalterego.research.dense_programs import prepare_fold, score_programs
from scripts.hyser_differential_program_pilot import candidate_programs, fixed_program


def test_adjacent_pair_map_and_common_mode_cancellation():
    names = tuple(f"{grid}-{row}-{col}" for grid in ("AA", "BB", "CC", "DD")
                  for row in range(1, 9) for col in range(1, 9))
    pairs = adjacent_pairs(names)
    assert len(pairs) == 448
    assert all(names[left][:2] == names[right][:2] for left, right in pairs)
    rng = np.random.default_rng(8)
    signal = rng.normal(size=(100, 256)).astype(np.float32)
    baseline = differential_trial_features(signal, pairs)
    common = np.sin(np.linspace(0, 2, len(signal))).astype(np.float32)
    shifted = differential_trial_features(signal + common[:, None], pairs)
    np.testing.assert_allclose(baseline, shifted, atol=1e-6)


def test_batched_ridge_accepts_448_pair_channels():
    rng = np.random.default_rng(9)
    train_x = rng.normal(size=(54, 896))
    test_x = rng.normal(size=(27, 896))
    train_y = np.repeat([1, 2, 3], 18)
    test_y = np.repeat([1, 2, 3], 9)
    selected = np.array([[400, 401]], dtype=np.int16)
    full = score_programs([prepare_fold((train_x, train_y), (test_x, test_y),
                                        features_per_site=2)], selected,
                          features_per_site=2)
    columns = [800, 801, 802, 803]
    reduced = score_programs([prepare_fold((train_x[:, columns], train_y),
                                           (test_x[:, columns], test_y),
                                           features_per_site=2)],
                             np.array([[0, 1]], dtype=np.int16),
                             features_per_site=2)
    np.testing.assert_allclose(full, reduced)


def test_sampled_program_counts_contacts_and_channels():
    names = tuple(f"{grid}-{row}-{col}" for grid in ("AA", "BB", "CC", "DD")
                  for row in range(1, 9) for col in range(1, 9))
    pairs = adjacent_pairs(names)
    metadata = {"site_names": names, "pairs": pairs}
    programs = candidate_programs(metadata, 20)
    programs = np.vstack((programs, fixed_program(metadata)))
    for program in programs:
        assert len(program) == 8
        assert len({contact for channel in program for contact in pairs[channel]}) == 12
