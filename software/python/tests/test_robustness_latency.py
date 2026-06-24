"""Tests for motion gating, latency benchmark, and online preprocess extensions."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from openalterego.dsp.online import OnlinePreprocessor
from openalterego.dsp.quality import fast_chunk_motion_index
from openalterego.ml.model import OpenAlterEgoCNN
from openalterego.runtime.latency_benchmark import run_latency_benchmark


class TestMotionGate(unittest.TestCase):
    def test_fast_chunk_motion_index_range(self) -> None:
        fs = 250.0
        t = np.arange(500) / fs
        clean = np.sin(2 * np.pi * 40 * t).astype(np.float32)[:, None]
        drift = (np.linspace(-1, 1, 500).astype(np.float32)[:, None] * 50.0)
        mi_clean = fast_chunk_motion_index(clean, fs)
        mi_drift = fast_chunk_motion_index(drift, fs)
        self.assertGreaterEqual(mi_clean, 0.0)
        self.assertLessEqual(mi_clean, 1.0)
        self.assertGreater(mi_drift, mi_clean)

    def test_online_preprocessor_motion_gate_attenuates(self) -> None:
        fs = 250.0
        pre = OnlinePreprocessor(
            fs_hz=fs,
            channels=2,
            motion_gate=True,
            motion_threshold=0.05,
            motion_attenuation=0.0,
        )
        t = np.arange(200) / fs
        x = (np.linspace(-1, 1, 200).astype(np.float32)[:, None] * np.ones((1, 2)))
        y = pre.process(x)
        self.assertTrue(pre.last_motion_gated)
        self.assertAlmostEqual(float(np.max(np.abs(y))), 0.0, places=5)


class TestLatencyBenchmark(unittest.TestCase):
    def test_latency_benchmark_runs(self) -> None:
        labels = ["yes", "no"]
        m = OpenAlterEgoCNN(channels=4, classes=len(labels))
        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "m.pt"
            torch.save(
                {
                    "state_dict": m.state_dict(),
                    "labels": labels,
                    "fs": 250,
                    "channels": 4,
                    "preprocess_mode": "streaming",
                    "emg_mode": "standard",
                    "segment_ms": 600,
                },
                ckpt,
            )
            rep = run_latency_benchmark(
                model_path=str(ckpt),
                n_chunks=30,
                warmup_chunks=5,
                window_ms=400,
                stride_ms=100,
            )
            self.assertGreater(rep.preprocess.n, 0)
            self.assertGreater(rep.inference_window.n, 0)
            self.assertGreater(rep.preprocess.p50_ms, 0.0)
            self.assertTrue(any("approx_e2e" in n for n in rep.notes))


if __name__ == "__main__":
    unittest.main()
