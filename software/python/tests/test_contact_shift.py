import numpy as np
import pytest

from openalterego.research.contact_shift import (
    align_shifted_features, channel_shift_maps, robust_options,
    shift_programs)
from openalterego.research.hyser_differential import adjacent_pairs
from scripts.hyser_compositional_search import edge_mode_options
from scripts.hyser_mixed_program_pilot import physical_contacts


def test_topological_shift_preserves_physical_program_and_rejects_boundary():
    names = tuple(f"{grid}-{row}-{col}" for grid in ("AA", "BB", "CC", "DD")
                  for row in range(1, 9) for col in range(1, 9))
    pairs = adjacent_pairs(names)
    meta = {"site_names": names, "pairs": pairs}
    maps = channel_shift_maps(meta)
    assert np.array_equal(maps[0], np.arange(maps.shape[1]))
    options = robust_options(edge_mode_options(meta), maps)
    assert all(0 < len(group) < 336 for group in options)
    program = np.asarray([[channel for group in options
                           for channel in group[0]]], dtype=np.int16)
    shifted = shift_programs(program, maps)
    assert shifted.shape == (5, 1, 8)
    for variant in shifted[:, 0]:
        assert len(physical_contacts(variant, pairs)) == 8
    raw = np.arange(maps.shape[1] * 2, dtype=np.float32).reshape(1, -1)
    aligned = align_shifted_features(raw, maps[1], features_per_channel=2)
    for channel in program[0]:
        assert np.array_equal(aligned[0, 2 * channel:2 * channel + 2],
                              raw[0, 2 * maps[1, channel]:2 * maps[1, channel] + 2])
    assert np.all(aligned.reshape(1, -1, 2)[0, maps[1] < 0] == 0)
    with pytest.raises(ValueError):
        shift_programs(np.asarray([[0]], dtype=np.int16), maps)
