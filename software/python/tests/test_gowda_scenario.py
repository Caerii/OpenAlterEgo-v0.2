"""Tests for Gowda-shaped biophysical scenario."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from openalterego.sim.dataset import generate_dataset
from openalterego.sim.scenarios.gowda_small_vocab import (
    build_gowda_dataset_config,
    build_gowda_schedule,
    load_gowda_labels,
)


class TestGowdaScenario(unittest.TestCase):
    def test_schedule_four_words_per_trial(self) -> None:
        sched = build_gowda_schedule(3, seed=1)
        self.assertEqual(len(sched), 12)
        self.assertEqual(sched[0].word_idx, 0)
        self.assertEqual(sched[4].trial_id, 1)
        self.assertEqual(sched[4].word_idx, 0)

    def test_labels_load(self) -> None:
        labs = load_gowda_labels()
        self.assertGreaterEqual(len(labs), 100)

    def test_generate_gowda_dataset_writes_trial_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ds = build_gowda_dataset_config(
                Path(tmp) / "gowda_sim",
                n_trials=4,
                seed=42,
                realism="off",
                snr_target_db=None,
                snr_motion_target_db=None,
            )
            ds.duration_s = 50.0
            ds.biophysical.band_limit_output = False  # type: ignore[union-attr]
            ds.biophysical.line_noise_uV = 0.0  # type: ignore[union-attr]
            out = generate_dataset(ds)
            ev = pd.read_csv(out / "events.csv")
            self.assertIn("trial_id", ev.columns)
            self.assertIn("word_idx", ev.columns)
            sig = np.load(out / "signals.npy")
            self.assertEqual(sig.shape[1], 31)
            self.assertTrue((out / "phonemes.csv").is_file())
            self.assertFalse(ev["trial_id"].isna().any())
            self.assertEqual(ev["trial_id"].nunique(), 4)
            self.assertGreaterEqual(len(ev), 4 * 4 - 1)

    def test_scripted_schedule_no_extra_events(self) -> None:
        """Biophysical stream must not emit random events after scripted queue drains."""
        with tempfile.TemporaryDirectory() as tmp:
            ds = build_gowda_dataset_config(
                Path(tmp) / "gowda_sim",
                n_trials=2,
                seed=7,
                realism="off",
                snr_target_db=None,
                snr_motion_target_db=None,
            )
            ds.duration_s = 30.0
            ds.biophysical.band_limit_output = False  # type: ignore[union-attr]
            ds.biophysical.line_noise_uV = 0.0  # type: ignore[union-attr]
            out = generate_dataset(ds)
            ev = pd.read_csv(out / "events.csv")
            self.assertFalse(ev["trial_id"].isna().any())
            self.assertLessEqual(len(ev), 8)
            self.assertGreaterEqual(len(ev), 6)
            self.assertEqual(ev["trial_id"].nunique(), 2)


if __name__ == "__main__":
    unittest.main()
