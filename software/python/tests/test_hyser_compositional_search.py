from openalterego.research.hyser_differential import adjacent_pairs
from scripts.hyser_compositional_search import edge_mode_options
from scripts.hyser_mixed_program_pilot import physical_contacts


def test_edge_mode_grammar_has_exact_physical_budget():
    names = tuple(f"{grid}-{row}-{col}" for grid in ("AA", "BB", "CC", "DD")
                  for row in range(1, 9) for col in range(1, 9))
    pairs = adjacent_pairs(names)
    options = edge_mode_options({"site_names": names, "pairs": pairs})
    assert tuple(len(group) for group in options) == (336, 336, 336, 336)
    for variant in range(3):
        program = tuple(sorted(channel for group in options
                               for channel in group[variant]))
        assert len(program) == 8
        assert len(physical_contacts(program, pairs)) == 8
