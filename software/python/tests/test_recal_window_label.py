"""Tests for re-calibration monitor, window sweep, and BLE labeling."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from openalterego.ml.model import OpenAlterEgoCNN
from openalterego.runtime.window_sweep import run_window_sweep
from openalterego.users.labeling import label_session_from_markers, markers_to_events
from openalterego.users.recalibration import RecalibrationMonitor, assess_session_recalibration


class TestRecalibration(unittest.TestCase):
    def test_monitor_sustained_deficit(self) -> None:
        mon = RecalibrationMonitor.from_baseline(20.0, warn_db=3.0, cooldown_s=0.0)
        st1 = mon.update(14.0, motion_index=0.1, now=1.0)
        st2 = mon.update(14.0, motion_index=0.1, now=2.0)
        st3 = mon.update(14.0, motion_index=0.1, now=3.0)
        self.assertFalse(st1.re_calibration_suggested)
        self.assertTrue(st3.re_calibration_suggested)
        self.assertTrue(st3.should_broadcast)

    def test_offline_session_check(self) -> None:
        st = assess_session_recalibration(
            session_snr_db=14.0,
            baseline_snr_db=20.0,
            motion_index=0.2,
            warn_db=3.0,
        )
        self.assertTrue(st.re_calibration_suggested)
        self.assertGreater(st.snr_deficit_db, 3.0)


class TestLabeling(unittest.TestCase):
    def test_time_s_markers(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            markers = Path(td) / "markers.csv"
            pd.DataFrame({"time_s": [0.5, 1.0], "label": ["yes", "no"]}).to_csv(markers, index=False)
            ev = markers_to_events(markers, fs_hz=250.0)
            self.assertEqual(len(ev), 2)
            self.assertEqual(int(ev.iloc[0]["start_sample"]), 125)

    def test_label_session_writes_events(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            session = Path(td) / "sess"
            session.mkdir()
            np.save(session / "signals.npy", np.zeros((1000, 4), dtype=np.float32))
            (session / "meta.json").write_text('{"fs_hz": 250}', encoding="utf-8")
            markers = Path(td) / "m.csv"
            pd.DataFrame({"sample": [100, 400], "label": ["a", "b"]}).to_csv(markers, index=False)
            out = label_session_from_markers(session, markers)
            self.assertTrue(out.is_file())
            ev = pd.read_csv(out)
            self.assertEqual(len(ev), 2)


class TestWindowSweep(unittest.TestCase):
    def test_window_sweep_runs(self) -> None:
        labels = ["yes", "no"]
        m = OpenAlterEgoCNN(channels=4, classes=2)
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            ckpt = td_path / "m.pt"
            torch.save(
                {
                    "state_dict": m.state_dict(),
                    "labels": labels,
                    "fs": 250,
                    "channels": 4,
                    "preprocess_mode": "streaming",
                    "emg_mode": "standard",
                },
                ckpt,
            )
            sess = td_path / "sess"
            sess.mkdir()
            np.save(sess / "signals.npy", np.random.randn(2000, 4).astype(np.float32) * 10)
            pd.DataFrame(
                {"start_sample": [200, 800], "end_sample": [450, 1050], "label": ["yes", "no"]}
            ).to_csv(sess / "events.csv", index=False)
            rep = run_window_sweep(
                model_path=str(ckpt),
                session_dir=sess,
                window_values_ms=[400, 600],
                n_latency_chunks=20,
            )
            self.assertEqual(len(rep.rows), 2)
            self.assertIsNotNone(rep.rows[0].event_accuracy)


if __name__ == "__main__":
    unittest.main()
