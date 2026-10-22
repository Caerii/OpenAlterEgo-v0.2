import numpy as np
import pytest

from scripts.hyser_differential_tie_audit import tied_maximum


def test_tied_maximum_uses_fixed_tolerance_and_canonical_order():
    scores = np.array([0.8, 0.9, 0.9 + 1e-13, 0.89])
    np.testing.assert_array_equal(tied_maximum(scores), [1, 2])
    with pytest.raises(ValueError):
        tied_maximum(np.array([np.nan]))
