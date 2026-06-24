"""Literature-aligned synthetic EMG band resolution and sim metadata."""

from __future__ import annotations

import unittest

from openalterego.sim.literature import (
    PARADIGM_ALTEREGO,
    PARADIGM_SEMG_CLAMPED,
    PARADIGM_SEMG_FULL,
    resolve_sim_token_band,
)
from openalterego.sim.stream import ScenarioConfig, SimStream, SimStreamConfig


class TestResolveSimTokenBand(unittest.TestCase):
    def test_clamped_at_250hz(self) -> None:
        lo, hi = resolve_sim_token_band(250, PARADIGM_SEMG_CLAMPED, None)
        self.assertAlmostEqual(lo, 20.0)
        self.assertAlmostEqual(hi, 115.0)  # Nyquist 125 - 10

    def test_alterego_envelope_at_250hz(self) -> None:
        lo, hi = resolve_sim_token_band(250, PARADIGM_ALTEREGO, None)
        self.assertAlmostEqual(lo, 1.0)
        self.assertAlmostEqual(hi, 50.0)

    def test_full_band_requires_high_fs(self) -> None:
        with self.assertRaises(ValueError):
            resolve_sim_token_band(250, PARADIGM_SEMG_FULL, None)
        lo, hi = resolve_sim_token_band(1000, PARADIGM_SEMG_FULL, None)
        self.assertAlmostEqual(lo, 20.0)
        self.assertAlmostEqual(hi, 450.0)

    def test_explicit_override_still_clamps_nyquist(self) -> None:
        lo, hi = resolve_sim_token_band(200, PARADIGM_SEMG_CLAMPED, (20.0, 450.0))
        self.assertAlmostEqual(lo, 20.0)
        self.assertAlmostEqual(hi, 90.0)  # 100 - 10


class TestSimStreamLiteratureMeta(unittest.TestCase):
    def test_chunk_meta_includes_paradigm(self) -> None:
        cfg = SimStreamConfig(
            fs_hz=250,
            channels=4,
            seed=3,
            realtime_clock=False,
            scenario=ScenarioConfig(labels=["a"], p_event=0.0),
            emg_paradigm=PARADIGM_SEMG_CLAMPED,
            ar1_innovation_scale=0.0,
        )
        sim = SimStream(cfg)
        ch = sim.next_chunk()
        self.assertEqual(ch.meta.get("sim_emg_paradigm"), PARADIGM_SEMG_CLAMPED)
        self.assertIn("sim_token_band_hz", ch.meta)
        self.assertEqual(len(ch.meta["sim_token_band_hz"]), 2)
        self.assertEqual(ch.meta.get("sim_literature_model"), "openalterego_emg_sim_v1")


if __name__ == "__main__":
    unittest.main()
