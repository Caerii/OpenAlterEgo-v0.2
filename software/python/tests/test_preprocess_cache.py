"""Preprocess disk cache tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from openalterego.dsp.preprocess_cache import (
    build_session_preprocess_cache,
    ensure_preprocessed_signals,
    is_cache_valid,
    load_cached_signals,
)
from openalterego.ml.datasets.session import SessionMeta, write_session_folder


class TestPreprocessCache(unittest.TestCase):
    def _tiny_session(self, td: Path) -> Path:
        sig = np.random.randn(4000, 4).astype(np.float32)
        ev = pd.DataFrame(
            {
                "start_sample": [100, 1200],
                "end_sample": [1000, 2200],
                "label": ["a", "b"],
            }
        )
        sess = td / "sess"
        write_session_folder(sess, sig, ev, SessionMeta(fs_hz=250.0, channels=4, source="test"))
        return sess

    def test_build_and_hit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            sess = self._tiny_session(Path(td))
            rep1 = build_session_preprocess_cache(
                sess, preprocess_mode="streaming", emg_mode="standard", show_progress=False
            )
            self.assertTrue(rep1.built)
            rep2 = build_session_preprocess_cache(
                sess, preprocess_mode="streaming", emg_mode="standard", show_progress=False
            )
            self.assertFalse(rep2.built)
            cached = load_cached_signals(
                sess, preprocess_mode="streaming", emg_mode="standard", fs_hz=250.0
            )
            self.assertIsNotNone(cached)
            self.assertEqual(cached.shape, (4000, 4))

    def test_invalidates_on_raw_change(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            sess = self._tiny_session(td_path)
            build_session_preprocess_cache(
                sess, preprocess_mode="streaming", emg_mode="standard", show_progress=False
            )
            npy_path, meta_path = (
                sess / "preprocess_cache" / "streaming_standard_250hz.npy",
                sess / "preprocess_cache" / "streaming_standard_250hz.meta.json",
            )
            self.assertTrue(is_cache_valid(meta_path, sess / "signals.npy", preprocess_mode="streaming", emg_mode="standard", fs_hz=250))
            raw = np.load(sess / "signals.npy")
            raw[0, 0] += 1.0
            np.save(sess / "signals.npy", raw)
            self.assertFalse(
                is_cache_valid(meta_path, sess / "signals.npy", preprocess_mode="streaming", emg_mode="standard", fs_hz=250)
            )

    def test_ensure_used_by_train_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            sess = self._tiny_session(Path(td))
            raw = np.load(sess / "signals.npy")
            out1, hit1 = ensure_preprocessed_signals(
                raw, sess, preprocess_mode="streaming", emg_mode="standard", fs_hz=250, show_progress=False
            )
            out2, hit2 = ensure_preprocessed_signals(
                raw, sess, preprocess_mode="streaming", emg_mode="standard", fs_hz=250, show_progress=False
            )
            self.assertFalse(hit1)
            self.assertTrue(hit2)
            self.assertEqual(out1.shape, out2.shape)


if __name__ == "__main__":
    unittest.main()
