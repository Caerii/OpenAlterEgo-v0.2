"""Performance benchmarks and scaling-law validation for biophysical synthesis."""

from __future__ import annotations

import unittest

import numpy as np

from openalterego.sim.biophysical.benchmark import (
    auto_chunk_ms_for_fs,
    benchmark_chunk,
    recommend_chunk_ms,
    run_extended_scaling_sweep,
    run_scaling_sweep,
)
from openalterego.sim.biophysical.accel_backend import active_backend_label, resolve_backend
from openalterego.sim.biophysical.pool_numba import HAS_NUMBA, scatter_unit_multichannel
from openalterego.sim.biophysical.pool_fast import MotorPoolSynthCache, superpose_motor_pool_fast
from openalterego.sim.biophysical.stream import BiophysicalSimStream, BiophysicalSimStreamConfig
from openalterego.sim.stream import ScenarioConfig


class TestPoolFast(unittest.TestCase):
    def test_convolve_path_finite(self) -> None:
        rng = np.random.default_rng(7)
        n, c = 500, 8
        w = rng.random((12, c)).astype(np.float32)
        w /= w.sum(axis=1, keepdims=True)
        tpl = [np.sin(np.linspace(0, 4, 15)).astype(np.float32) for _ in range(12)]
        cache = MotorPoolSynthCache.from_pool(tpl, w, channel_delays=None)
        x = np.zeros((n, c), dtype=np.float32)
        rates = rng.uniform(20.0, 80.0, size=12)
        amps = rng.uniform(30.0, 60.0, size=12)
        n_sp = superpose_motor_pool_fast(
            x, 500.0, rng, rates, amps, cache, spread_across_channels=True
        )
        self.assertGreater(n_sp, 0)
        self.assertTrue(np.isfinite(x).all())
        self.assertGreater(float(np.std(x)), 0.0)


class TestBenchmark(unittest.TestCase):
    def test_benchmark_chunk_runs(self) -> None:
        t = benchmark_chunk(fs_hz=250, channels=4, n_motor_units=24, n_chunks=20, warmup=2)
        self.assertGreater(t.samples_per_s, 0.0)
        self.assertGreater(t.realtime_factor, 0.0)

    def test_fast_mode_high_throughput_at_1khz(self) -> None:
        t_fast = benchmark_chunk(
            fs_hz=1000, channels=8, n_motor_units=48, chunk_ms=80,
            synth_mode="fast", n_chunks=40, warmup=3,
            use_conduction_delays=False,
        )
        self.assertGreater(t_fast.realtime_factor, 20.0)
        self.assertGreater(t_fast.samples_per_s, 100_000.0)

    def test_auto_chunk_ms_in_valid_range(self) -> None:
        for fs in (250, 500, 1000):
            ms = auto_chunk_ms_for_fs(fs)
            self.assertGreaterEqual(ms, 20)
            self.assertLessEqual(ms, 200)
        ms, t = recommend_chunk_ms(500, target_realtime_factor=5.0)
        self.assertGreaterEqual(ms, 20)
        self.assertLessEqual(ms, 200)
        self.assertGreater(t.realtime_factor, 0.0)

    def test_scaling_sweep_compact(self) -> None:
        rep = run_scaling_sweep(
            fs_values=[250, 500],
            channel_values=[8],
            mu_values=[48],
            chunk_ms=40,
        )
        self.assertGreaterEqual(len(rep.timings), 3)
        self.assertGreater(rep.recommended_chunk_ms, 0)

    def test_recommend_chunk_ms(self) -> None:
        ms, t = recommend_chunk_ms(500, target_realtime_factor=5.0)
        self.assertGreaterEqual(ms, 20)
        self.assertLessEqual(ms, 200)
        self.assertGreater(t.realtime_factor, 0.0)


class TestNumbaKernels(unittest.TestCase):
    @unittest.skipUnless(HAS_NUMBA, "numba not installed")
    def test_numba_scatter_finite(self) -> None:
        rng = np.random.default_rng(3)
        n, c = 200, 8
        x = np.zeros((n, c), dtype=np.float32)
        idx = np.array([10, 50, 120], dtype=np.int32)
        env = np.ones((n,), dtype=np.float32)
        m = np.sin(np.linspace(0, 3, 12)).astype(np.float32)
        wi = np.ones((c,), dtype=np.float32) / float(c)
        scatter_unit_multichannel(x, idx, 40.0, env, m, wi, n, c, m.size)
        self.assertTrue(np.isfinite(x).all())
        self.assertGreater(float(np.std(x)), 0.0)

    @unittest.skipUnless(HAS_NUMBA, "numba not installed")
    def test_numba_mode_throughput(self) -> None:
        t = benchmark_chunk(
            fs_hz=1000, channels=8, n_motor_units=48, chunk_ms=80,
            synth_mode="numba", n_chunks=30, warmup=3,
            use_conduction_delays=True,
        )
        self.assertGreater(t.realtime_factor, 15.0)


class TestAccelBackend(unittest.TestCase):
    def test_resolve_auto(self) -> None:
        backend = resolve_backend("auto")
        self.assertIn(backend, ("python", "numba", "rust"))

    def test_active_backend_label(self) -> None:
        self.assertIn(active_backend_label(), ("python", "numba", "rust"))


class TestExtendedSweep(unittest.TestCase):
    def test_extended_sweep_runs(self) -> None:
        rep = run_extended_scaling_sweep(
            chunk_ms=40,
            synth_mode="fast",
            use_conduction_delays=False,
        )
        self.assertGreaterEqual(len(rep.timings), 9)
        self.assertTrue(any("backend=" in n for n in rep.notes))


class TestStreamFastMode(unittest.TestCase):
    def test_fast_stream_produces_events(self) -> None:
        cfg = BiophysicalSimStreamConfig(
            fs_hz=500,
            channels=8,
            seed=2,
            synth_mode="fast",
            scenario=ScenarioConfig(labels=["a", "b"], p_event=0.9),
            realtime_clock=False,
            use_conduction_delays=False,
        )
        sim = BiophysicalSimStream(cfg)
        for _ in range(50):
            ch = sim.next_chunk()
            self.assertTrue(np.isfinite(ch.samples).all())
        self.assertGreater(len(sim.events), 0)


if __name__ == "__main__":
    unittest.main()
