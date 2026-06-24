"""Segment tensor disk cache tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from openalterego.ml.datasets.session import SessionMeta, write_session_folder
from openalterego.ml.segment_cache import (
    build_segment_arrays,
    ensure_segment_arrays,
    is_segment_cache_valid,
    load_segment_cache,
    segment_cache_paths,
    segment_cache_stem,
    write_segment_cache,
)


class TestSegmentCache(unittest.TestCase):
    def _tiny_session(self, td: Path) -> tuple[Path, pd.DataFrame, dict[str, int]]:
        sig = np.random.randn(4000, 4).astype(np.float32)
        ev = pd.DataFrame(
            {
                "start_sample": [100, 1200, 2400],
                "end_sample": [1000, 2200, 3600],
                "label": ["a", "b", "a"],
            }
        )
        label_to_id = {"a": 0, "b": 1}
        sess = td / "sess"
        write_session_folder(sess, sig, ev, SessionMeta(fs_hz=250.0, channels=4, source="test"))
        return sess, ev, label_to_id

    def test_build_segment_arrays_shape(self) -> None:
        sig = np.random.randn(3000, 4).astype(np.float32)
        ev = pd.DataFrame(
            {"start_sample": [0, 500], "end_sample": [400, 1200], "label": ["x", "y"]}
        )
        X, y = build_segment_arrays(
            sig, ev, {"x": 0, "y": 1}, fs_hz=250, segment_ms=600, seed=0
        )
        self.assertEqual(X.shape[0], 2)
        self.assertEqual(X.shape[1], 4)
        self.assertEqual(X.shape[2], 150)  # 250 * 600 / 1000
        self.assertEqual(list(y), [0, 1])

    def test_write_load_and_hit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            sess, ev, label_to_id = self._tiny_session(Path(td))
            sig = np.load(sess / "signals.npy")
            stem = segment_cache_stem(
                preprocess_mode="streaming",
                emg_mode="standard",
                fs_hz=250,
                segment_ms=600,
                seed=7,
                split_tag="tr",
                channel_indices=None,
                events=ev,
            )
            X, y = build_segment_arrays(
                sig, ev, label_to_id, fs_hz=250, segment_ms=600, seed=7
            )
            write_segment_cache(
                sess,
                stem,
                X,
                y,
                preprocess_mode="streaming",
                emg_mode="standard",
                fs_hz=250,
                segment_ms=600,
                seed=7,
                split_tag="tr",
                channel_indices=None,
                events=ev,
            )
            npz_path, meta_path = segment_cache_paths(sess, stem)
            self.assertTrue(npz_path.is_file())
            self.assertTrue(meta_path.is_file())
            loaded = load_segment_cache(
                sess,
                stem,
                preprocess_mode="streaming",
                emg_mode="standard",
                fs_hz=250,
                segment_ms=600,
                seed=7,
                split_tag="tr",
                channel_indices=None,
                events=ev,
            )
            self.assertIsNotNone(loaded)
            X2, y2 = loaded  # type: ignore[misc]
            np.testing.assert_allclose(X2, X, rtol=1e-5, atol=1e-5)
            np.testing.assert_array_equal(y2, y)

    def test_ensure_hit_after_first_build(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            sess, ev, label_to_id = self._tiny_session(Path(td))
            sig = np.load(sess / "signals.npy")
            _, _, hit1 = ensure_segment_arrays(
                sig,
                ev,
                label_to_id,
                sess,
                preprocess_mode="streaming",
                emg_mode="standard",
                fs_hz=250,
                segment_ms=600,
                seed=1,
                split_tag="tr",
            )
            _, _, hit2 = ensure_segment_arrays(
                sig,
                ev,
                label_to_id,
                sess,
                preprocess_mode="streaming",
                emg_mode="standard",
                fs_hz=250,
                segment_ms=600,
                seed=1,
                split_tag="tr",
            )
            self.assertFalse(hit1)
            self.assertTrue(hit2)

    def test_invalidates_on_event_change(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            sess, ev, label_to_id = self._tiny_session(Path(td))
            sig = np.load(sess / "signals.npy")
            stem = segment_cache_stem(
                preprocess_mode="streaming",
                emg_mode="standard",
                fs_hz=250,
                segment_ms=600,
                seed=0,
                split_tag="tr",
                channel_indices=None,
                events=ev,
            )
            X, y = build_segment_arrays(
                sig, ev, label_to_id, fs_hz=250, segment_ms=600, seed=0
            )
            write_segment_cache(
                sess,
                stem,
                X,
                y,
                preprocess_mode="streaming",
                emg_mode="standard",
                fs_hz=250,
                segment_ms=600,
                seed=0,
                split_tag="tr",
                channel_indices=None,
                events=ev,
            )
            _, meta_path = segment_cache_paths(sess, stem)
            self.assertTrue(
                is_segment_cache_valid(
                    meta_path,
                    preprocess_meta_path=sess / "missing.meta.json",
                    preprocess_mode="streaming",
                    emg_mode="standard",
                    fs_hz=250,
                    segment_ms=600,
                    seed=0,
                    split_tag="tr",
                    channel_indices=None,
                    events=ev,
                )
            )
            ev2 = ev.copy()
            ev2.loc[0, "label"] = "c"
            self.assertFalse(
                is_segment_cache_valid(
                    meta_path,
                    preprocess_meta_path=sess / "missing.meta.json",
                    preprocess_mode="streaming",
                    emg_mode="standard",
                    fs_hz=250,
                    segment_ms=600,
                    seed=0,
                    split_tag="tr",
                    channel_indices=None,
                    events=ev2,
                )
            )


    def test_per_event_preprocess_changes_output(self) -> None:
        sig = np.random.randn(5000, 4).astype(np.float32)
        ev = pd.DataFrame({"start_sample": [0, 2000], "end_sample": [2000, 4000], "label": ["a", "b"]})
        l2i = {"a": 0, "b": 1}
        X0, _ = build_segment_arrays(sig, ev, l2i, fs_hz=250, segment_ms=400, seed=0, per_event_preprocess=False)
        X1, _ = build_segment_arrays(sig, ev, l2i, fs_hz=250, segment_ms=400, seed=0, per_event_preprocess=True, preprocess_emg_mode="standard")
        self.assertEqual(X0.shape, X1.shape)
        self.assertFalse(np.allclose(X0, X1))


if __name__ == "__main__":
    unittest.main()
