"""End-to-end and cross-module integration checks."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from openalterego.dsp.emg_config import emg_signal_band_hz_for_quality
from openalterego.ml.infer import load_model
from openalterego.ml.model import OpenAlterEgoCNN
from openalterego.users.calibration import CalibrationConfig, calibrate_user
from openalterego.users.manager import UserManager
from openalterego.users.profile import UserProfile


def _require_torch() -> None:
    if torch is None:
        raise unittest.SkipTest("torch not installed")


def _write_min_session(path: Path, *, fs_hz: int = 250, n_per_class: int = 25) -> None:
    rng = np.random.default_rng(99)
    seg_samples = max(150, int(fs_hz * 600 / 1000))
    rows: list[dict] = []
    chunks: list[np.ndarray] = []
    t = 0
    for label, ch in [("yes", 0), ("no", 1)]:
        for _ in range(n_per_class):
            block = rng.standard_normal((seg_samples, 4)).astype(np.float32) * 0.3
            block[:, ch] += 2.0
            chunks.append(block)
            rows.append({"start_sample": t, "end_sample": t + seg_samples, "label": label})
            t += seg_samples + 20
    signals = np.concatenate(chunks, axis=0)
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "signals.npy", signals)
    pd.DataFrame(rows).to_csv(path / "events.csv", index=False)


class TestCheckpointUserId(unittest.TestCase):
    def test_load_model_reads_user_id(self) -> None:
        _require_torch()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "m.pt"
            m = OpenAlterEgoCNN(channels=2, classes=2)
            torch.save(
                {
                    "state_dict": m.state_dict(),
                    "labels": ["a", "b"],
                    "fs": 100,
                    "channels": 2,
                    "preprocess_mode": "none",
                    "segment_ms": 10,
                    "user_id": "alice",
                },
                p,
            )
            lm = load_model(p)
            self.assertEqual(lm.user_id, "alice")

    def test_calibrate_writes_user_id_in_checkpoint(self) -> None:
        _require_torch()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = root / "sess"
            _write_min_session(session, n_per_class=20)
            mgr = UserManager(root / "users")
            mgr.save_profile(UserProfile(user_id="bob"))
            cfg = CalibrationConfig(
                min_samples_per_token=5,
                epochs=1,
                batch_size=16,
                val_split=0.2,
                seed=3,
            )
            calibrate_user("bob", session, fs_hz=250, user_manager=mgr, config=cfg, preprocessing_mode="standard")
            ckpt = torch.load(mgr.get_user_dir("bob") / "model.pt", map_location="cpu")
            self.assertEqual(ckpt.get("user_id"), "bob")


class TestEmgQualityBands(unittest.TestCase):
    def test_standard_clamped_to_nyquist(self) -> None:
        lo, hi = emg_signal_band_hz_for_quality("standard", 250.0)
        self.assertEqual(lo, 1.0)
        self.assertEqual(hi, 50.0)

    def test_clinical_and_wide(self) -> None:
        c_lo, c_hi = emg_signal_band_hz_for_quality("clinical", 250.0)
        self.assertEqual(c_lo, 0.5)
        w_lo, w_hi = emg_signal_band_hz_for_quality("wide", 1000.0)
        self.assertEqual(w_lo, 20.0)
        self.assertLessEqual(w_hi, 450.0)


if __name__ == "__main__":
    unittest.main()
