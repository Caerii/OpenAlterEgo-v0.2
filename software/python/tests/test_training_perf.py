"""Training throughput helper tests."""

from __future__ import annotations

import os
import unittest

import torch
from torch.utils.data import TensorDataset

from openalterego.ml.training_perf import (
    build_train_dataloader,
    default_num_workers,
    maybe_compile_model,
    resolve_use_amp,
)


class TestTrainingPerf(unittest.TestCase):
    def test_default_num_workers_explicit(self) -> None:
        self.assertEqual(default_num_workers(0), 0)
        self.assertEqual(default_num_workers(4), 4)

    def test_default_num_workers_auto_bounded(self) -> None:
        nw = default_num_workers(-1)
        self.assertGreaterEqual(nw, 0)
        self.assertLessEqual(nw, 8)
        if os.name == "nt":
            self.assertEqual(nw, 0)

    def test_build_train_dataloader_cpu(self) -> None:
        ds = TensorDataset(torch.zeros(8, 2), torch.zeros(8, dtype=torch.long))
        dl = build_train_dataloader(ds, batch_size=4, device=torch.device("cpu"), num_workers=0)
        batch = next(iter(dl))
        self.assertEqual(batch[0].shape[0], 4)

    def test_maybe_compile_disabled_returns_same(self) -> None:
        m = torch.nn.Linear(4, 2)
        out = maybe_compile_model(m, enabled=False)
        self.assertIs(out, m)

    def test_resolve_use_amp_cpu_default_off(self) -> None:
        self.assertFalse(resolve_use_amp(torch.device("cpu")))
        self.assertFalse(resolve_use_amp(torch.device("cpu"), no_amp=False))

    def test_resolve_use_amp_cuda_default_on(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self.assertTrue(resolve_use_amp(torch.device("cuda")))
        self.assertFalse(resolve_use_amp(torch.device("cuda"), no_amp=True))


if __name__ == "__main__":
    unittest.main()
