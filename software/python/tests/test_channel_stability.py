import numpy as np
import pytest

from openalterego.research.channel_stability import class_contrast


def test_class_contrast_ignores_shared_recording_offset():
    x = np.asarray([[0., 1., 2., 3.], [1., 2., 3., 4.],
                    [3., 4., 5., 6.], [4., 5., 6., 7.]])
    labels = np.asarray([0, 0, 1, 1])
    contrast, within = class_contrast(x, labels, classes=2, channels=2)
    shifted, shifted_within = class_contrast(
        x + np.asarray([10., 20., 30., 40.]), labels,
        classes=2, channels=2)
    np.testing.assert_allclose(contrast, shifted)
    np.testing.assert_allclose(within, shifted_within)


def test_class_contrast_requires_every_class():
    with pytest.raises(ValueError, match="channel contrast"):
        class_contrast(np.ones((3, 4)), np.zeros(3, dtype=int),
                       classes=2, channels=2)
