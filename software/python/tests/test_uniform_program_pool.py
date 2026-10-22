import numpy as np
import pytest

from openalterego.research.uniform_program_pool import uniform_program_prefix


def test_sampling_preserves_nested_prefix_and_unique_complete_programs():
    options = tuple(tuple((offset + i,) for i in range(5))
                    for offset in (0, 10, 20))
    short = uniform_program_prefix(options, 20, 42)
    long = uniform_program_prefix(options, 50, 42)
    np.testing.assert_array_equal(short, long[:20])
    assert len({tuple(row) for row in long}) == 50
    assert long.shape == (50, 3)


def test_sampling_rejects_more_than_feasible_space():
    with pytest.raises(ValueError, match="not enough unique"):
        uniform_program_prefix((((0,), (1,)),), 3, 42)
