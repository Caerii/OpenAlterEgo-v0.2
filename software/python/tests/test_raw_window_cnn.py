import numpy as np
import pytest
import torch

from openalterego.research.raw_window_cnn import RawWindowCNN, normalize_window_batch


def test_window_normalization_preserves_shape_and_rejects_bad_input():
    rng = np.random.default_rng(42)
    windows = rng.normal(size=(4, 400, 14)).astype(np.float32)
    normalized = normalize_window_batch(windows, (0, 2, 5, 8, 11, 13))
    assert normalized.shape == (4, 6, 400)
    assert np.isfinite(normalized).all()
    with pytest.raises(ValueError):
        normalize_window_batch(windows[:, :399], (0, 2, 5, 8, 11, 13))


def test_cnn_forward_has_one_logit_per_event_and_class():
    model = RawWindowCNN()
    assert model(torch.zeros(3, 6, 400)).shape == (3, 9)
