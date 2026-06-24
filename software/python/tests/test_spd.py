"""Tests for SPD edge matrices and σ(τ) features."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from openalterego.ml.spd.basis import fit_spd_basis, save_spd_basis, load_spd_basis
from openalterego.ml.spd.features import (
    GOWDA_SPD_ETA,
    edge_matrix,
    segment_to_sigma_sequence,
    spd_regularize,
    sigma_tau,
    sliding_edge_matrices,
)


class TestSPDFeatures(unittest.TestCase):
    def test_edge_matrix_symmetric(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.standard_normal((200, 8)).astype(np.float64)
        e = edge_matrix(x)
        self.assertEqual(e.shape, (8, 8))
        np.testing.assert_allclose(e, e.T, atol=1e-10)

    def test_spd_regularize_increases_trace_fraction(self) -> None:
        x = np.eye(4, dtype=np.float64)
        x[0, 1] = 0.5
        x[1, 0] = 0.5
        reg = spd_regularize(x, eta=GOWDA_SPD_ETA)
        self.assertGreater(float(np.min(np.linalg.eigvalsh(reg))), 0.0)

    def test_sigma_sequence_shape(self) -> None:
        rng = np.random.default_rng(1)
        seg = rng.standard_normal((10000, 31)).astype(np.float32)
        mats = sliding_edge_matrices(seg, fs_hz=5000, window_ms=100, step_ms=50)
        basis = fit_spd_basis(mats[:20], channels=31, fs_hz=5000)
        sig = segment_to_sigma_sequence(seg, basis.basis_q, fs_hz=5000)
        self.assertEqual(sig.shape[1], 31 * 31)
        self.assertGreater(sig.shape[0], 30)

    def test_basis_cache_roundtrip(self) -> None:
        rng = np.random.default_rng(2)
        mats = [spd_regularize(edge_matrix(rng.standard_normal((250, 31)))) for _ in range(5)]
        basis = fit_spd_basis(mats, channels=31, fs_hz=5000)
        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp)
            save_spd_basis(session, "test_basis", basis)
            loaded = load_spd_basis(session, "test_basis")
            np.testing.assert_allclose(loaded.basis_q, basis.basis_q, atol=1e-8)
            self.assertEqual(loaded.feature_dim, 31 * 31)


class TestSPDCTCCollate(unittest.TestCase):
    def test_spd_dataset_builds(self) -> None:
        from openalterego.ml.ctc.dataset import PhonemeCTCDataset
        from openalterego.ml.spd.basis import fit_spd_basis_from_events

        rng = np.random.default_rng(3)
        n = 50000
        ch = 8
        signals = rng.standard_normal((n, ch)).astype(np.float32)
        events = pd.DataFrame(
            {
                "start_sample": [0, 10000],
                "end_sample": [10000, 20000],
                "label": ["monday", "tuesday"],
                "trial_id": [0, 1],
                "word_idx": [0, 0],
            }
        )
        basis = fit_spd_basis_from_events(signals, events.iloc[:1], fs_hz=5000, max_matrices=50)
        ds = PhonemeCTCDataset(
            signals,
            events,
            fs_hz=5000,
            segment_ms=2000,
            seed=1,
            feature_type="spd",
            spd_basis=basis,
        )
        self.assertEqual(len(ds), 2)
        x, ph, plen = ds[0]
        self.assertEqual(x.ndim, 2)
        self.assertGreater(int(ph.shape[0]), plen - 1)


if __name__ == "__main__":
    unittest.main()
