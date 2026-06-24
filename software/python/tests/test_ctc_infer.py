"""Tests for CTC checkpoint loading."""

from __future__ import annotations

import unittest
from pathlib import Path


class TestCTCInfer(unittest.TestCase):
    def test_load_phase5_checkpoint_if_present(self) -> None:
        ckpt = Path("sessions/gowda_sv_full/ablations/ctc_spd_v3_diag_delta_seed1337.pt")
        if not ckpt.is_file():
            self.skipTest("phase5 checkpoint not present")
        from openalterego.ml.ctc.infer import load_ctc_model

        loaded = load_ctc_model(ckpt, device_preferred="cpu", session_dir=ckpt.parent.parent)
        self.assertTrue(loaded.uses_spd)
        self.assertIsNotNone(loaded.basis_q)
        self.assertEqual(loaded.feature_mode, "diag_delta")


if __name__ == "__main__":
    unittest.main()
