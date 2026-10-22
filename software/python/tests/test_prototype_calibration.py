import numpy as np

from openalterego.research.prototype_calibration import adapt_centroids, predict_nearest


def test_calibration_shrinks_new_class_and_leaves_unobserved_class_unchanged():
    prior = np.array([[0.0], [10.0]])
    updated = adapt_centroids(prior, np.array([[5.0], [5.0]]),
                              np.array([0, 0]), pseudocount=2)
    np.testing.assert_allclose(updated, [[2.5], [10.0]])
    np.testing.assert_array_equal(predict_nearest(np.array([[2.0], [9.0]]), updated),
                                  [0, 1])
