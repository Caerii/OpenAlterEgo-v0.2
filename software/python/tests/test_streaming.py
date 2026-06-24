"""Tests for realtime streaming decode / stabilizer."""

from __future__ import annotations

import unittest

from openalterego.runtime.streaming import PredictionStabilizer, StreamDecodeConfig


class TestPredictionStabilizer(unittest.TestCase):
    def test_stability_requires_n_matching(self) -> None:
        cfg = StreamDecodeConfig(stable_n=2, min_confidence=0.5)
        s = PredictionStabilizer(cfg)
        self.assertIsNone(s.update("a", 0.9, t=0.0, seq=0, source="x"))
        self.assertIsNone(s.update("b", 0.9, t=0.01, seq=1, source="x"))
        ev = s.update("a", 0.9, t=0.02, seq=2, source="x")
        self.assertIsNone(ev)
        ev2 = s.update("a", 0.9, t=0.03, seq=3, source="x")
        self.assertIsNotNone(ev2)
        self.assertEqual(ev2.token, "a")

    def test_adaptive_threshold_lowers_gate(self) -> None:
        cfg = StreamDecodeConfig(
            stable_n=1,
            min_confidence=1.0,
            adaptive_threshold=True,
            threshold_alpha=0.95,
            threshold_clip_low=0.5,
            threshold_clip_high=0.95,
        )
        s = PredictionStabilizer(cfg)
        ev = None
        for i in range(40):
            ev = s.update("tok", 0.99, t=float(i) * 0.01, seq=i, source="s")
            if ev is not None:
                break
        self.assertIsNotNone(ev)
        self.assertLess(s.effective_threshold(), 0.99)

    def test_snr_deficit_raises_gate(self) -> None:
        cfg = StreamDecodeConfig(
            stable_n=1,
            min_confidence=0.5,
            baseline_snr_db=20.0,
            snr_deficit_scale=0.05,
            snr_gate_cap=0.4,
        )
        s = PredictionStabilizer(cfg)
        ev_ok = s.update("x", 0.85, t=0.0, seq=0, source="s", snr_db=20.0)
        self.assertIsNotNone(ev_ok)
        s.reset()
        ev_bad = s.update("x", 0.75, t=0.0, seq=1, source="s", snr_db=5.0)
        self.assertIsNone(ev_bad)

    def test_reset_clears_state(self) -> None:
        cfg = StreamDecodeConfig(stable_n=2, min_confidence=0.5, adaptive_threshold=True, threshold_alpha=0.5)
        s = PredictionStabilizer(cfg)
        s.update("a", 0.99, t=0.0, seq=0, source="x")
        s.reset()
        self.assertEqual(s.effective_threshold(), 0.5)
        self.assertIsNone(s.update("a", 0.99, t=0.01, seq=1, source="x"))

    def test_abstain_clears_history_no_emit(self) -> None:
        """Abstention clears debouncer state so a run of stable_n is required again."""
        cfg = StreamDecodeConfig(stable_n=3, min_confidence=0.5)
        s = PredictionStabilizer(cfg)
        s.update("a", 0.99, t=0.0, seq=0, source="x")
        s.update("a", 0.99, t=0.01, seq=1, source="x")
        self.assertIsNone(s.update("a", 0.99, t=0.02, seq=2, source="x", abstain=True))
        self.assertIsNone(s.update("a", 0.99, t=0.03, seq=3, source="x"))
        self.assertIsNone(s.update("a", 0.99, t=0.04, seq=4, source="x"))
        ev = s.update("a", 0.99, t=0.05, seq=5, source="x")
        self.assertIsNotNone(ev)
        self.assertEqual(ev.token, "a")


if __name__ == "__main__":
    unittest.main()
