"""Tests for sim→real transfer merge and split."""

from __future__ import annotations

import json
import gc
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from openalterego.ml.datasets.events import sanitize_trial_events
from openalterego.ml.eval.sim_transfer import _merge_sim_real_train
from openalterego.ml.spd.basis import ensure_gowda_spd_basis


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
    def test_basis_fit_reads_source_and_targets_external_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source, cache = Path(tmp) / "real", Path(tmp) / "cache"
            source.mkdir()
            np.save(source / "signals.npy", np.zeros((100, 31), dtype=np.float32))
            pd.DataFrame({
                "start_sample": [0], "end_sample": [50], "label": ["monday"],
                "trial_id": [0], "word_idx": [0],
            }).to_csv(source / "events.csv", index=False)
            fitted = SimpleNamespace(
                basis_q=np.eye(31, dtype=np.float32), channels=31, fs_hz=5000,
                window_ms=100, step_ms=50, eta=1e-6, n_train_matrices=1,
            )
            with patch("openalterego.ml.spd.basis.load_spd_basis", side_effect=FileNotFoundError), patch(
                "openalterego.ml.spd.basis.fit_spd_basis_from_events", return_value=fitted
            ) as fit, patch("openalterego.ml.spd.basis.save_spd_basis") as save:
                ensure_gowda_spd_basis(source, basis_cache_dir=cache, fs_hz=5000)
            self.assertEqual(fit.call_args.args[0].shape, (100, 31))
            self.assertEqual(save.call_args.args[0], cache)
            self.assertEqual(sorted(p.name for p in source.iterdir()), ["events.csv", "signals.npy"])
            fit.reset_mock()
            gc.collect()  # release the captured memmap before Windows temp cleanup

    def test_merge_uses_external_workspace_and_only_real_train_trials(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sim_dir, real_dir, work_dir = root / "sim", root / "real", root / "work"
            sim_dir.mkdir()
            real_dir.mkdir()
            np.save(sim_dir / "signals.npy", np.zeros((100, 31), dtype=np.float32))
            real_signals = np.zeros((100, 31), dtype=np.float32)
            real_signals[20:30] = 99.0  # held-out trial must not enter the merged file
            np.save(real_dir / "signals.npy", real_signals)
            columns = ["start_sample", "end_sample", "label", "trial_id", "word_idx"]
            pd.DataFrame([[0, 10, "monday", 0, 0]], columns=columns).to_csv(sim_dir / "events.csv", index=False)
            pd.DataFrame(
                [[0, 10, "monday", 0, 0], [20, 30, "tuesday", 400, 0]], columns=columns
            ).to_csv(real_dir / "events.csv", index=False)
            (sim_dir / "meta.json").write_text('{"fs_hz": 5000}', encoding="utf-8")
            out = _merge_sim_real_train(sim_dir, real_dir, 1.0, seed=1, work_dir=work_dir)
            self.assertEqual(out.parent, work_dir)
            self.assertFalse((real_dir / "ablations").exists())
            events = pd.read_csv(out / "events.csv")
            self.assertEqual(len(events), 2)
            self.assertNotIn("tuesday", events["label"].tolist())
            merged_signals = np.load(out / "signals.npy")
            self.assertEqual(merged_signals.shape, (110, 31))
            self.assertEqual(float(merged_signals.max()), 0.0)

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
