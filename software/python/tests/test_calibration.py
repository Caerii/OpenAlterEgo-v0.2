"""Tests for user calibration (requires torch)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from openalterego.users.calibration import CalibrationConfig, calibrate_user
from openalterego.users.manager import UserManager


def _require_torch() -> None:
    if torch is None:
        raise unittest.SkipTest("torch is not installed")


def _write_synthetic_session(path: Path, *, fs_hz: int, channels: int = 4, n_per_class: int = 40) -> None:
    """Minimal session with separable channel bias per class."""
    rng = np.random.default_rng(42)
    seg_samples = max(150, int(fs_hz * 600 / 1000))
    rows: list[dict] = []
    chunks: list[np.ndarray] = []
    t = 0
    for label, ch in [("yes", 0), ("no", 1)]:
        for _ in range(n_per_class):
            block = rng.standard_normal((seg_samples, channels)).astype(np.float32) * 0.3
            block[:, ch] += 2.0
            chunks.append(block)
            rows.append(
                {
                    "start_sample": t,
                    "end_sample": t + seg_samples,
                    "label": label,
                }
            )
            t += seg_samples + 20
    signals = np.concatenate(chunks, axis=0)
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "signals.npy", signals)
    pd.DataFrame(rows).to_csv(path / "events.csv", index=False)


class TestCalibrateWideValidation(unittest.TestCase):
    """Runs without PyTorch (calibration defers torch import until training)."""

    def test_wide_mode_low_fs_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "session"
            _write_synthetic_session(data_dir, fs_hz=250)
            mgr = UserManager(tmp_path / "userdata")
            cfg = CalibrationConfig(min_samples_per_token=5, epochs=1, batch_size=8)
            with self.assertRaises(ValueError) as ctx:
                calibrate_user(
                    "bob",
                    data_dir,
                    fs_hz=250,
                    user_manager=mgr,
                    config=cfg,
                    preprocessing_mode="wide",
                )
            self.assertIn("Wide mode requires", str(ctx.exception))

    def test_strict_motion_rejects_high_motion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "session"
            _write_synthetic_session(data_dir, fs_hz=250)
            mgr = UserManager(tmp_path / "userdata")
            fake = mock.MagicMock()
            fake.snr_db = 30.0
            fake.motion_index = 0.99
            cfg = CalibrationConfig(
                min_samples_per_token=5,
                epochs=1,
                batch_size=8,
                strict_motion=True,
                motion_reject_above=0.2,
            )
            with mock.patch("openalterego.users.calibration.assess_signal_quality", return_value=fake):
                with self.assertRaises(ValueError) as ctx:
                    calibrate_user(
                        "bob",
                        data_dir,
                        fs_hz=250,
                        user_manager=mgr,
                        config=cfg,
                        preprocessing_mode="standard",
                    )
            self.assertIn("strict_motion", str(ctx.exception))


class TestCalibrateUser(unittest.TestCase):
    def setUp(self) -> None:
        _require_torch()

    def test_calibrate_writes_artifacts_and_streaming_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "session"
            _write_synthetic_session(data_dir, fs_hz=250)
            mgr = UserManager(tmp_path / "userdata")
            cfg = CalibrationConfig(
                min_samples_per_token=5,
                val_split=0.25,
                epochs=2,
                batch_size=16,
                segment_ms=600,
                seed=42,
            )
            profile, report = calibrate_user(
                "alice",
                data_dir,
                fs_hz=250,
                user_manager=mgr,
                config=cfg,
                preprocessing_mode="standard",
            )

            udir = mgr.get_user_dir("alice")
            self.assertTrue((udir / "model.pt").is_file())
            self.assertTrue((udir / "profile.json").is_file())
            self.assertTrue((udir / "calibration_report.json").is_file())
            self.assertTrue((udir / "calibration_report.txt").is_file())

            self.assertEqual(profile.user_id, "alice")
            self.assertEqual(profile.preprocessing_mode, "standard")
            self.assertTrue(0.5 <= profile.confidence_threshold <= 0.95)
            self.assertIsNotNone(profile.model_path)

            ckpt = torch.load(udir / "model.pt", map_location="cpu")
            self.assertEqual(ckpt["preprocess_mode"], "streaming")
            self.assertEqual(ckpt["emg_mode"], "standard")
            self.assertEqual(ckpt["segment_ms"], 600)
            self.assertEqual(ckpt["fs"], 250)
            self.assertIn("yes", ckpt["labels"])
            self.assertIn("no", ckpt["labels"])
            self.assertEqual(profile.stride_ms, 120)

            self.assertEqual(report.user_id, "alice")
            self.assertGreaterEqual(len(report.tokens), 2)
            with open(udir / "calibration_report.json", encoding="utf-8") as f:
                meta = json.load(f)
            self.assertEqual(meta["user_id"], "alice")
            self.assertIn("confidence_threshold", meta)

    def test_calibrate_clinical_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "session"
            _write_synthetic_session(data_dir, fs_hz=250, n_per_class=25)
            mgr = UserManager(tmp_path / "userdata")
            cfg = CalibrationConfig(
                min_samples_per_token=5,
                val_split=0.2,
                epochs=2,
                batch_size=16,
                seed=7,
            )
            profile, _ = calibrate_user(
                "carol",
                data_dir,
                fs_hz=250,
                user_manager=mgr,
                config=cfg,
                preprocessing_mode="clinical",
            )
            self.assertEqual(profile.preprocessing_mode, "clinical")
            ckpt = torch.load(mgr.get_user_dir("carol") / "model.pt", map_location="cpu")
            self.assertEqual(ckpt["emg_mode"], "clinical")

    def test_calibrate_stride_ms_saved_on_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "session"
            _write_synthetic_session(data_dir, fs_hz=250)
            mgr = UserManager(tmp_path / "userdata")
            cfg = CalibrationConfig(
                min_samples_per_token=5,
                val_split=0.25,
                epochs=1,
                batch_size=16,
                segment_ms=600,
                stride_ms=80,
                seed=11,
            )
            profile, _ = calibrate_user(
                "stride_user",
                data_dir,
                fs_hz=250,
                user_manager=mgr,
                config=cfg,
                preprocessing_mode="standard",
            )
            self.assertEqual(profile.stride_ms, 80)
            with open(mgr.get_profile_path("stride_user"), encoding="utf-8") as f:
                blob = json.load(f)
            self.assertEqual(blob["stride_ms"], 80)

    def test_calibrate_without_stratified_split_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "session"
            _write_synthetic_session(data_dir, fs_hz=250)
            mgr = UserManager(tmp_path / "userdata")
            cfg = CalibrationConfig(
                min_samples_per_token=5,
                val_split=0.25,
                epochs=1,
                batch_size=16,
                stratified_split=False,
                seed=99,
            )
            profile, report = calibrate_user(
                "shuffle_val",
                data_dir,
                fs_hz=250,
                user_manager=mgr,
                config=cfg,
                preprocessing_mode="standard",
            )
            self.assertTrue(profile.model_path.is_file())
            self.assertGreaterEqual(len(report.tokens), 2)

    def test_default_user_manager_uses_cwd_users(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "session"
            _write_synthetic_session(data_dir, fs_hz=250, n_per_class=20)
            cfg = CalibrationConfig(
                min_samples_per_token=5,
                epochs=1,
                batch_size=16,
                seed=1,
            )
            with mock.patch("openalterego.users.calibration.Path.cwd", return_value=tmp_path):
                profile, _ = calibrate_user(
                    "dave",
                    data_dir,
                    fs_hz=250,
                    user_manager=None,
                    config=cfg,
                    preprocessing_mode="standard",
                )
            expected_model = tmp_path / "users" / "dave" / "model.pt"
            self.assertTrue(expected_model.is_file())
            self.assertEqual(profile.user_id, "dave")


if __name__ == "__main__":
    unittest.main()
