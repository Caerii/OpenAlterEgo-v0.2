"""Group-level splits and channel analysis."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from openalterego.ml.analysis import run_channel_importance
from openalterego.ml.data_split import (
    resolve_train_val_indices,
    stratified_group_train_val_indices,
)
from openalterego.ml.datasets.session import SessionMeta, write_session_folder
from openalterego.ml.model import create_model


class TestGroupSplit(unittest.TestCase):
    def test_no_trial_leakage(self) -> None:
        labels = np.array(["a", "a", "b", "b", "c", "c", "c", "c"])
        groups = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        tr, va = stratified_group_train_val_indices(labels, groups, 0.25, seed=0)
        tr_g = set(groups[tr])
        va_g = set(groups[va])
        self.assertTrue(tr_g.isdisjoint(va_g))

    def test_auto_uses_trial_id(self) -> None:
        labels = np.array(["x"] * 4 + ["y"] * 4)
        groups = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        tr, va = resolve_train_val_indices(labels, 0.25, 0, split_by="auto", groups=groups)
        self.assertTrue(set(groups[tr]).isdisjoint(set(groups[va])))


class TestChannelImportance(unittest.TestCase):
    def test_runs_on_tiny_session(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            sig = np.random.randn(2000, 8).astype(np.float32)
            ev = pd.DataFrame(
                {
                    "start_sample": [100, 600, 1100],
                    "end_sample": [500, 1000, 1500],
                    "label": ["yes", "no", "yes"],
                }
            )
            write_session_folder(
                td_path / "sess",
                sig,
                ev,
                SessionMeta(fs_hz=250.0, channels=8, source="test"),
            )
            model = create_model("se_resnet", channels=8, classes=2)
            ckpt = {
                "state_dict": model.state_dict(),
                "labels": ["no", "yes"],
                "fs": 250,
                "channels": 8,
                "preprocess_mode": "streaming",
                "emg_mode": "standard",
                "segment_ms": 400,
                "arch": "se_resnet",
            }
            ckpt_path = td_path / "model.pt"
            torch.save(ckpt, ckpt_path)
            rep = run_channel_importance(td_path / "sess", ckpt_path, segment_ms=400, top_k=4)
            self.assertEqual(len(rep.combined_scores), 8)
            self.assertEqual(len(rep.top_channels), 4)


if __name__ == "__main__":
    unittest.main()
