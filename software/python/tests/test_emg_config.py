"""EMG mode resolution and wide-band guards."""

from __future__ import annotations

import unittest

import numpy as np

from openalterego.dsp.emg_config import resolve_emg_mode_for_serve
from openalterego.dsp.filters import preprocess_streaming


class TestResolveEmgModeForServe(unittest.TestCase):
    def test_checkpoint_wins_over_profile(self) -> None:
        m = resolve_emg_mode_for_serve(
            checkpoint_emg_mode="clinical",
            profile_preprocessing_mode="wide",
        )
        self.assertEqual(m, "clinical")

    def test_profile_when_no_ckpt_field(self) -> None:
        m = resolve_emg_mode_for_serve(
            checkpoint_emg_mode=None,
            profile_preprocessing_mode="wide",
        )
        self.assertEqual(m, "wide")

    def test_default_standard(self) -> None:
        m = resolve_emg_mode_for_serve(
            checkpoint_emg_mode=None,
            profile_preprocessing_mode=None,
        )
        self.assertEqual(m, "standard")


class TestPreprocessStreamingWideFs(unittest.TestCase):
    def test_wide_rejects_low_fs(self) -> None:
        x = np.zeros((64, 2), dtype=np.float32)
        with self.assertRaises(ValueError):
            preprocess_streaming(x, fs_hz=250.0, channels=2, mode="wide")


if __name__ == "__main__":
    unittest.main()
