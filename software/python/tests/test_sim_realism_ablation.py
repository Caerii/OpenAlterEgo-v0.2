"""Realism ablation harness (probe + optional transfer)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from openalterego.ml.eval.sim_realism_ablation import run_probe_ablation
from openalterego.sim.metrics.realism_match import (
    RealismVariant,
    default_realism_variants,
    match_score,
    parse_variant_tags,
    sim_gowda_variant_stats,
)


class TestRealismAblation(unittest.TestCase):
    def test_default_variant_tags_unique(self) -> None:
        tags = [v.tag for v in default_realism_variants()]
        self.assertEqual(len(tags), len(set(tags)))

    def test_parse_variant_tags_subset(self) -> None:
        got = parse_variant_tags(["off_raw", "tang_cal"])
        self.assertEqual([v.tag for v in got], ["off_raw", "tang_cal"])

    def test_sim_probe_stats_finite(self) -> None:
        variant = RealismVariant("tang_cal", preset="tang", snr_target_db=18.9, snr_motion_target_db=12.7)
        with tempfile.TemporaryDirectory() as td:
            stats, extra = sim_gowda_variant_stats(variant, probe_trials=2, seed=7, out_dir=Path(td))
        self.assertGreater(stats.n_segments, 0)
        self.assertIsNotNone(stats.median_snr_db)
        self.assertIn("snr_calibration", extra["meta"])

    def test_match_score_lower_for_self(self) -> None:
        variant = RealismVariant("off_raw", preset="off")
        with tempfile.TemporaryDirectory() as td:
            stats, _ = sim_gowda_variant_stats(variant, probe_trials=2, seed=3, out_dir=Path(td))
        self_score = match_score(stats, stats)["total"]
        other = RealismVariant("field_cal", preset="field", snr_target_db=18.9, snr_motion_target_db=12.7)
        with tempfile.TemporaryDirectory() as td2:
            other_stats, _ = sim_gowda_variant_stats(other, probe_trials=2, seed=3, out_dir=Path(td2))
        cross = match_score(other_stats, stats)["total"]
        self.assertLessEqual(self_score, cross)


class TestRealismProbeOnReal(unittest.TestCase):
    def test_probe_ablation_if_real_session_present(self) -> None:
        real = Path(__file__).resolve().parents[1] / "sessions" / "gowda_sv_full"
        if not (real / "signals.npy").is_file():
            self.skipTest("gowda_sv_full session not present")
        report = run_probe_ablation(
            real,
            [RealismVariant("off_raw", preset="off"), RealismVariant("tang_cal", preset="tang", snr_target_db=18.9, snr_motion_target_db=12.7)],
            probe_trials=2,
            seed=11,
        )
        self.assertEqual(len(report["variants"]), 2)
        self.assertIn("match", report["variants"][0])


if __name__ == "__main__":
    unittest.main()
