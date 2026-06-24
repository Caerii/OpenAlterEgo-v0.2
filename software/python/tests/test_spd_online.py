"""Tests for online SPD stream vs offline segment path."""

from __future__ import annotations

import unittest

import numpy as np

from openalterego.dsp.filters import preprocess_basic
from openalterego.ml.spd.features import segment_to_sigma_sequence
from openalterego.ml.spd.online import OnlineSPDStream


class TestOnlineSPD(unittest.TestCase):
    def test_build_sequence_matches_offline(self) -> None:
        rng = np.random.default_rng(0)
        c, t = 8, 2000
        q = np.eye(c, dtype=np.float64)
        seg = rng.standard_normal((t, c)).astype(np.float32)
        stream = OnlineSPDStream(q, fs_hz=5000, channels=c, feature_mode="diag_delta", emg_mode="gowda")
        online = stream.build_sequence(seg)
        seg_pp = preprocess_basic(seg, fs_hz=5000, mode="gowda", rectify_signals=False, normalize_mode="zscore")
        offline = segment_to_sigma_sequence(seg_pp, q, fs_hz=5000, feature_mode="diag_delta")
        self.assertEqual(online.shape, offline.shape)
        np.testing.assert_allclose(online, offline, rtol=1e-4, atol=1e-4)

    def test_push_accumulates_frames(self) -> None:
        rng = np.random.default_rng(1)
        c = 8
        q = np.eye(c, dtype=np.float64)
        stream = OnlineSPDStream(q, fs_hz=5000, channels=c, feature_mode="diag", emg_mode="gowda")
        chunk = rng.standard_normal((500, c)).astype(np.float32)
        frames = stream.push(chunk)
        self.assertGreater(len(frames), 0)


if __name__ == "__main__":
    unittest.main()
