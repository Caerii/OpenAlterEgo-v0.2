import numpy as np

from openalterego.research.hyser_differential import adjacent_pairs
from scripts.hyser_mixed_program_pilot import (
    fixed_mixed, mixed_programs, physical_contacts)


def test_mixed_program_uses_eight_array_contacts_and_eight_afe_channels():
    names = tuple(f"{grid}-{row}-{col}" for grid in ("AA", "BB", "CC", "DD")
                  for row in range(1, 9) for col in range(1, 9))
    pairs = adjacent_pairs(names)
    meta = {"site_names": names, "pairs": pairs}
    programs = np.vstack((mixed_programs(meta, 20), fixed_mixed(meta)))
    for program in programs:
        mono = [int(channel) for channel in program if channel < 256]
        differential = [int(channel) - 256 for channel in program if channel >= 256]
        assert len(mono) == len(differential) == 4
        contacts = set(mono)
        for pair_index in differential:
            contacts.update(pairs[pair_index])
            assert pairs[pair_index][0] in mono
        assert len(contacts) == 8
        assert physical_contacts(program, pairs) == tuple(sorted(contacts))
