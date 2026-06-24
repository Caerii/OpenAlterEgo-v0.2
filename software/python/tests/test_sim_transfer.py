"""Tests for sim→real transfer merge and split."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from openalterego.ml.datasets.events import sanitize_trial_events
from openalterego.ml.eval.sim_transfer import _merge_sim_real_train


class TestSimTransferSanitize(unittest.TestCase):
    def test_sanitize_drops_nan_trial_id(self) -> None:
        ev = pd.DataFrame(
            {
                "start_sample": [0, 100],
                "end_sample": [50, 150],
                "label": ["a", "b"],
                "trial_id": [0.0, np.nan],
                "word_idx": [0.0, np.nan],
            }
        )
        out = sanitize_trial_events(ev)
        self.assertEqual(len(out), 1)
        self.assertEqual(int(out.iloc[0]["trial_id"]), 0)


class TestSimTransferMergeIntegration(unittest.TestCase):
    def test_merge_writes_split_mode_meta(self) -> None:
        real_root = Path("sessions/gowda_sv_full")
        if not (real_root / "events.csv").is_file():
            self.skipTest("gowda_sv_full not present")

        with tempfile.TemporaryDirectory() as tmp:
            sim_dir = Path(tmp) / "sim"
            sim_dir.mkdir()
            np.save(sim_dir / "signals.npy", np.zeros((1000, 31), dtype=np.float32))
            pd.DataFrame(
                {
                    "start_sample": [0, 200],
                    "end_sample": [100, 300],
                    "label": ["monday", "january"],
                    "trial_id": [0, 0],
                    "word_idx": [0, 1],
                }
            ).to_csv(sim_dir / "events.csv", index=False)
            (sim_dir / "meta.json").write_text('{"fs_hz": 5000}', encoding="utf-8")

            out = _merge_sim_real_train(sim_dir, real_root, 0.1, seed=1)
            meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta.get("split_mode"), "sim_transfer_merged")
            ev = pd.read_csv(out / "events.csv")
            sim_trials = int(meta["sim_trials"])
            self.assertTrue((ev["trial_id"] >= sim_trials).any())
            self.assertFalse(ev["trial_id"].isna().any())


if __name__ == "__main__":
    unittest.main()
