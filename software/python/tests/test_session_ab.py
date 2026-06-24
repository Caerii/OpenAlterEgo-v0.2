"""Tests for session preprocessing A/B eval."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from openalterego.ml.datasets.session import SessionMeta, write_session_folder
from openalterego.ml.eval.session_ab import run_session_preprocess_ab


class TestSessionAB(unittest.TestCase):
    def test_ab_runs_on_synthetic_session(self) -> None:
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            labels = ["alpha", "beta", "gamma"]
            events = []
            chunks = []
            offset = 0
            for i, lab in enumerate(labels * 8):
                seg = rng.normal(0, 20, size=(400, 4)).astype(np.float32)
                start = offset + 50
                end = start + 400
                events.append({"start_sample": start, "end_sample": end, "label": lab})
                chunks.append(seg)
                offset = end + 50
            signals = np.vstack(chunks)
            sess = td_path / "sess"
            write_session_folder(
                sess,
                signals,
                pd.DataFrame(events),
                SessionMeta(fs_hz=1000.0, channels=4, source="test"),
            )
            rep = run_session_preprocess_ab(
                sess,
                emg_modes=["standard", "wide"],
                epochs=2,
                segment_ms=400,
                min_samples_per_label=2,
                show_progress=False,
            )
            self.assertEqual(len(rep.rows), 2)
            self.assertGreater(rep.rows[0].val_acc, 0.0)


if __name__ == "__main__":
    unittest.main()
