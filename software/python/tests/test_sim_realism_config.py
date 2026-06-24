"""SimConfig realism wiring."""

from __future__ import annotations

import unittest

from openalterego.acquisition.simulate import SimConfig, stream_simulated_chunks


class TestSimRealismPreset(unittest.TestCase):
    def test_invalid_realism_raises(self) -> None:
        cfg = SimConfig(realism_preset="not_a_mode", realtime_clock=False)
        with self.assertRaises(ValueError):
            next(stream_simulated_chunks(cfg))

    def test_off_sets_chunk_meta_for_biophysical(self) -> None:
        cfg = SimConfig(
            sim_engine="biophysical",
            realism_preset="off",
            channels=4,
            realtime_clock=False,
            labels=["yes"],
        )
        ch = next(stream_simulated_chunks(cfg))
        self.assertEqual(ch.meta.get("sim_realism_preset"), "off")


if __name__ == "__main__":
    unittest.main()
