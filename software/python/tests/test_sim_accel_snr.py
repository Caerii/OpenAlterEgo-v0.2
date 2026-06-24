"""Tests for batched spike events and dual-regime SNR calibration."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from openalterego.sim.biophysical.pool_events import collect_spike_events
from openalterego.sim.biophysical.pool_fast import MotorPoolSynthCache
from openalterego.sim.dataset import DatasetConfig, generate_dataset_shards
from openalterego.sim.snr_calibration import TANG_SNR_MOTION_DB, TANG_SNR_STATIC_DB, tune_snr_regimes
from openalterego.sim.stream import ScenarioConfig, SimStreamConfig


class TestPoolEvents(unittest.TestCase):
    def test_collect_spike_events_nonempty(self) -> None:
        rng = np.random.default_rng(1)
        n, c = 400, 8
        tpl = [np.sin(np.linspace(0, 3, 14)).astype(np.float32) for _ in range(6)]
        w = rng.random((6, c)).astype(np.float32)
        w /= w.sum(axis=1, keepdims=True)
        cache = MotorPoolSynthCache.from_pool(tpl, w, channel_delays=None)
        rates = rng.uniform(30.0, 90.0, size=6)
        amps = rng.uniform(40.0, 70.0, size=6)
        mu, t0, amp, total = collect_spike_events(
            rng, rates, amps, cache, n=n, fs_hz=500.0, envelope=None,
            time_jitter_std_s=0.0, refractory_samples=None,
        )
        self.assertGreater(total, 0)
        self.assertEqual(mu.shape[0], total)
        self.assertEqual(t0.shape[0], total)
        self.assertEqual(amp.shape[0], total)


class TestSnrRegimes(unittest.TestCase):
    def test_tune_snr_regimes_runs(self) -> None:
        cal = tune_snr_regimes(
            fs_hz=250,
            channels=4,
            realism_preset="tang",
            static_target_db=TANG_SNR_STATIC_DB,
            motion_target_db=TANG_SNR_MOTION_DB,
            probe_duration_s=4.0,
            max_iter=6,
            seed=3,
        )
        self.assertGreater(cal.lf_snr_scale, 0.0)
        self.assertGreater(cal.motion_burst_scale, 0.0)
        d = cal.to_dict()
        self.assertIn("static_snr_db", d)
        self.assertIn("motion_snr_db", d)


class TestDatasetShards(unittest.TestCase):
    def test_generate_two_shards(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            parent = Path(td) / "corpus"
            ds = DatasetConfig(
                out_dir=parent,
                duration_s=1.0,
                config=SimStreamConfig(
                    fs_hz=250,
                    channels=4,
                    seed=9,
                    scenario=ScenarioConfig(labels=["a", "b"], p_event=0.8),
                    realtime_clock=False,
                ),
                sim_engine="heuristic",
            )
            paths = generate_dataset_shards(ds, n_shards=2, workers=2)
            self.assertEqual(len(paths), 2)
            for p in paths:
                self.assertTrue((p / "signals.npy").is_file())
                self.assertTrue((p / "events.csv").is_file())


if __name__ == "__main__":
    unittest.main()
