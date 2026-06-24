"""Coarticulation envelope tests."""

from __future__ import annotations

import unittest

import numpy as np

from openalterego.sim.phonology.coarticulation import (
    build_phone_coarticulation_envelopes,
    coarticulation_overlap_ms,
    iter_coarticulated_phone_jobs,
)


class TestCoarticulation(unittest.TestCase):
    def test_envelope_shape_and_normalized(self) -> None:
        seg = [800, 500, 1200, 400]
        env = build_phone_coarticulation_envelopes(seg, min_overlap_samples=80)
        self.assertEqual(env.shape, (4, sum(seg)))
        col_sum = np.sum(env, axis=0)
        np.testing.assert_allclose(col_sum, np.ones_like(col_sum), atol=0.05)

    def test_overlap_at_boundaries(self) -> None:
        seg = [1000, 1000]
        env = build_phone_coarticulation_envelopes(seg, overlap_fraction=0.3, min_overlap_samples=100)
        mid = 1000
        self.assertGreater(float(env[0, mid - 50]), 0.05)
        self.assertGreater(float(env[1, mid + 49]), 0.05)

    def test_iter_jobs_cover_overlap_twice(self) -> None:
        seg = [500, 500]
        env = build_phone_coarticulation_envelopes(seg, min_overlap_samples=100)
        jobs = list(iter_coarticulated_phone_jobs(400, 200, seg, env))
        pids = {pid for _a, _b, pid, _ in jobs}
        self.assertIn(0, pids)
        self.assertIn(1, pids)

    def test_single_phone_no_coart_matrix(self) -> None:
        env = build_phone_coarticulation_envelopes([2000], min_overlap_samples=50)
        np.testing.assert_allclose(env, np.ones((1, 2000), dtype=np.float32))

    def test_overlap_ms_positive(self) -> None:
        seg = [500, 500]
        ms = coarticulation_overlap_ms(seg, 5000.0, min_overlap_ms=10.0)
        self.assertEqual(len(ms), 1)
        self.assertGreaterEqual(ms[0], 10.0)


if __name__ == "__main__":
    unittest.main()
