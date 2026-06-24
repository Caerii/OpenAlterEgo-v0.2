"""Tests for neurobiophysical motor pool (fiber types, batch synthesis)."""

from __future__ import annotations

import time
import unittest

import numpy as np

from openalterego.sim.biophysical.physiology import (
    FIBER_TYPE_SPECS,
    FiberType,
    init_physiological_motor_pool,
)
from openalterego.sim.biophysical.pool import add_muap_spikes
from openalterego.sim.biophysical.pool_batch import superpose_motor_pool_window
from openalterego.sim.biophysical.stream import BiophysicalSimStream, BiophysicalSimStreamConfig
from openalterego.sim.biophysical.versions import BIOPHYS_MODEL_VERSION
from openalterego.sim.stream import ScenarioConfig


class TestPhysiologicalPool(unittest.TestCase):
    def test_fiber_types_assigned(self) -> None:
        rng = np.random.default_rng(0)
        _lab, _w, gain, phys, tpl = init_physiological_motor_pool(
            rng, 60, 8, 4, fs_hz=250.0
        )
        self.assertEqual(phys.fiber_type.shape[0], 60)
        self.assertTrue(np.any(phys.fiber_type == int(FiberType.S)))
        self.assertTrue(np.any(phys.fiber_type == int(FiberType.FF)))
        self.assertEqual(len(tpl), 60)
        # S units should have wider templates than FF on average.
        s_lens = [tpl[i].size for i in range(60) if phys.fiber_type[i] == int(FiberType.S)]
        ff_lens = [tpl[i].size for i in range(60) if phys.fiber_type[i] == int(FiberType.FF)]
        if s_lens and ff_lens:
            self.assertGreater(float(np.mean(s_lens)), float(np.mean(ff_lens)))

    def test_max_rates_by_fiber_type(self) -> None:
        rng = np.random.default_rng(1)
        _lab, _w, _g, phys, _tpl = init_physiological_motor_pool(rng, 90, 4, 2, fs_hz=500.0)
        s_mask = phys.fiber_type == int(FiberType.S)
        ff_mask = phys.fiber_type == int(FiberType.FF)
        if np.any(s_mask) and np.any(ff_mask):
            self.assertLess(
                float(np.mean(phys.max_rate_hz[s_mask])),
                float(np.mean(phys.max_rate_hz[ff_mask])),
            )


class TestBatchSynthesis(unittest.TestCase):
    def test_batch_matches_legacy_finite(self) -> None:
        rng_a = np.random.default_rng(42)
        rng_b = np.random.default_rng(42)
        n, c = 250, 4
        w = np.array([[0.7, 0.2, 0.05, 0.05], [0.1, 0.8, 0.05, 0.05]], dtype=np.float32)
        w /= w.sum(axis=1, keepdims=True)
        m0 = np.sin(np.linspace(0, 6, 12)).astype(np.float32)
        m1 = np.cos(np.linspace(0, 5, 10)).astype(np.float32)
        rates = np.array([80.0, 40.0], dtype=np.float64)
        amps = np.array([50.0, 35.0], dtype=np.float64)

        xa = np.zeros((n, c), dtype=np.float32)
        xb = np.zeros((n, c), dtype=np.float32)
        for i in range(2):
            add_muap_spikes(
                xa, 250.0, rng_a, float(rates[i]), w[i], m0 if i == 0 else m1, float(amps[i]),
                spread_across_channels=True,
            )
        superpose_motor_pool_window(
            xb, 250.0, rng_b, rates, amps, w, [m0, m1], spread_across_channels=True
        )
        self.assertTrue(np.isfinite(xa).all())
        self.assertTrue(np.isfinite(xb).all())
        self.assertGreater(float(np.std(xa)), 0.0)
        self.assertGreater(float(np.std(xb)), 0.0)


class TestBiophysicalV5Stream(unittest.TestCase):
    def test_v5_model_version_in_meta(self) -> None:
        cfg = BiophysicalSimStreamConfig(
            fs_hz=250,
            channels=8,
            seed=3,
            scenario=ScenarioConfig(labels=["yes", "no"], p_event=0.8),
            realtime_clock=False,
            use_physiological_pool=True,
            use_batch_synthesis=True,
        )
        sim = BiophysicalSimStream(cfg)
        ch = sim.next_chunk()
        self.assertEqual(ch.meta.get("sim_biophysical_model"), BIOPHYS_MODEL_VERSION)
        self.assertEqual(BIOPHYS_MODEL_VERSION, "openalterego_biophysical_emg_v5")

    def test_batch_not_slower_than_legacy_order_of_magnitude(self) -> None:
        sc = ScenarioConfig(labels=["yes", "no"], p_event=0.9)
        cfg_batch = BiophysicalSimStreamConfig(
            fs_hz=250, channels=8, seed=5, scenario=sc, realtime_clock=False,
            use_physiological_pool=True, use_batch_synthesis=True, chunk_ms=40,
        )
        cfg_loop = BiophysicalSimStreamConfig(
            fs_hz=250, channels=8, seed=5, scenario=sc, realtime_clock=False,
            use_physiological_pool=True, use_batch_synthesis=False, chunk_ms=40,
        )
        sim_b = BiophysicalSimStream(cfg_batch)
        t0 = time.perf_counter()
        for _ in range(80):
            sim_b.next_chunk()
        tb = time.perf_counter() - t0

        sim_l = BiophysicalSimStream(cfg_loop)
        t1 = time.perf_counter()
        for _ in range(80):
            sim_l.next_chunk()
        tl = time.perf_counter() - t1
        # Batch path should not be dramatically slower (sanity, not strict perf test).
        self.assertLess(tb, tl * 3.0 + 0.5)


if __name__ == "__main__":
    unittest.main()
