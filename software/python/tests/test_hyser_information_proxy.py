import numpy as np
import pytest

from openalterego.research.hyser_information_proxy import (
    class_covariances, score_designs)


def test_label_signal_controls_information_and_batches_agree():
    rng = np.random.default_rng(4)
    labels = np.repeat([0, 1, 2], 20)
    features = rng.normal(size=(60, 4)) * 0.2
    features[:, :2] += labels[:, None] * 2.0
    between, within = class_covariances(features, labels)
    designs = np.array([[0], [1]], dtype=np.int16)
    fast = score_designs(between, within, designs, batch_size=2)
    slow = score_designs(between, within, designs, batch_size=1)
    np.testing.assert_allclose(fast, slow, atol=1e-12)
    assert fast[0] > fast[1]
    shuffled = class_covariances(features, labels[::-1])
    np.testing.assert_allclose(between, shuffled[0], atol=1e-12)
    with pytest.raises(ValueError):
        score_designs(between, within, np.array([[0, 0]]))
