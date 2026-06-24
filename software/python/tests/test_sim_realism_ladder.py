"""Realism preset ladder and SNR calibration (sim-to-real pragmatics)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from openalterego.sim.biophysical.sensor_pipeline import SensorNoiseState
from openalterego.sim.dataset import DatasetConfig, generate_dataset
from openalterego.sim.montage_geometry import (
    electrode_positions_1d,
    green_pickup_matrix_from_montage,
)
from openalterego.sim.realism import realism_preset_params
from openalterego.sim.snr_calibration import TANG_SNR_STATIC_DB, tune_noise_scale
from openalterego.sim.stream import ScenarioConfig, SimStreamConfig


class TestRealismLadder(unittest.TestCase):
    def test_preset_motion_escalates(self) -> None:
        off = realism_preset_params("off")
        wear = realism_preset_params("wearable")
        tang = realism_preset_params("tang")
        field = realism_preset_params("field")
        self.assertLess(off.motion_chunk_prob, wear.motion_chunk_prob)
        self.assertLess(wear.motion_chunk_prob, tang.motion_chunk_prob)
        self.assertLess(tang.motion_chunk_prob, field.motion_chunk_prob)
        self.assertGreater(tang.motion_shared_fraction, wear.motion_shared_fraction)

    def test_tang_preset_known(self) -> None:
        p = realism_preset_params("tang")
        self.assertGreater(p.contact_step_prob, 0.0)

    def test_sensor_noise_off_lt_tang(self) -> None:
        rng = np.random.default_rng(0)

        class _Cfg:
            electrode_noise_uV = 20.0
            ar1_phi = 0.97
            ar1_innovation_scale = 0.42
            line_noise_uV = 6.0
            mains_freq_hz = 60.0
            drift_uV_per_s = 12.0
            realism_preset = "off"

        off_state = SensorNoiseState(4, rng)
        tang_state = SensorNoiseState(4, rng)
        x_off = np.zeros((400, 4), dtype=np.float32)
        x_tang = np.zeros((400, 4), dtype=np.float32)
        _Cfg.realism_preset = "off"
        off_state.apply_chunk(x_off, _Cfg, rng, chunk_start=0, fs=250.0, noise_scale=1.0)
        _Cfg.realism_preset = "tang"
        tang_state.apply_chunk(x_tang, _Cfg, rng, chunk_start=0, fs=250.0, noise_scale=1.0)
        self.assertLess(float(np.std(x_off)), float(np.std(x_tang)))


class TestMontageGeometry(unittest.TestCase):
    def test_alterego_positions_ordered(self) -> None:
        pos = electrode_positions_1d("alterego_8ch")
        self.assertEqual(pos.shape[0], 8)
        self.assertTrue(float(pos.min()) >= 0.0)
        self.assertTrue(float(pos.max()) <= 1.0)

    def test_montage_pickup_differs_from_uniform(self) -> None:
        from openalterego.sim.biophysical.forward_model import green_pickup_matrix

        g_m = green_pickup_matrix_from_montage("alterego_8ch", 14)
        g_u = green_pickup_matrix(8, 14)
        self.assertEqual(g_m.shape, (8, 14))
        self.assertFalse(np.allclose(g_m, g_u, atol=0.02))


class TestSnrCalibration(unittest.TestCase):
    def test_tune_noise_scale_finite(self) -> None:
        cal = tune_noise_scale(
            fs_hz=250,
            channels=8,
            realism_preset="tang",
            target_snr_db=TANG_SNR_STATIC_DB,
            montage_name="alterego_8ch",
            probe_duration_s=4.0,
            seed=11,
        )
        self.assertGreater(cal.lf_snr_scale, 0.005)
        self.assertLess(cal.lf_snr_scale, 0.2)
        if cal.measured_snr_db is not None:
            self.assertLess(abs(cal.measured_snr_db - TANG_SNR_STATIC_DB), 2.5)

    def test_dataset_writes_snr_calibration_meta(self) -> None:
        sc = ScenarioConfig(labels=["yes", "no"], p_event=0.7)
        cfg = SimStreamConfig(
            fs_hz=250,
            channels=8,
            seed=3,
            scenario=sc,
            realtime_clock=False,
            line_noise_uV=6.0,
        )
        with tempfile.TemporaryDirectory() as td:
            ds = DatasetConfig(
                out_dir=Path(td),
                duration_s=3.0,
                config=cfg,
                sim_engine="biophysical",
                realism_preset="tang",
                snr_target_db=18.9,
                montage_name="alterego_8ch",
            )
            out = generate_dataset(ds)
            import json

            meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
            self.assertIn("snr_calibration", meta)
            self.assertIn("quality_metrics", meta)
            self.assertEqual(meta["sim_engine"], "biophysical")


if __name__ == "__main__":
    unittest.main()
