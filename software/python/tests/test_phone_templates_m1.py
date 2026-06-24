"""Tests for phoneme synthesizer M1 (durations, templates, fitting)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from openalterego.ml.phonology.phone_templates import fit_phone_templates
from openalterego.sim.phonology.durations import phone_duration_weights
from openalterego.sim.phonology.timeline import partition_event_by_weights, partition_phones_in_event
from openalterego.sim.phonology.templates import PhoneTemplate, PhoneTemplateStore, save_phone_templates


class TestPhoneDurations(unittest.TestCase):
    def test_vowel_longer_than_stop(self) -> None:
        w = phone_duration_weights(["AA", "T"])
        self.assertGreater(w[0], w[1])

    def test_weighted_partition_sums(self) -> None:
        rng = np.random.default_rng(0)
        seg = partition_event_by_weights(5000, [1.0, 0.35, 0.5], rng)
        self.assertEqual(len(seg), 3)
        self.assertEqual(sum(seg), 5000)

    def test_partition_phones_in_event(self) -> None:
        rng = np.random.default_rng(1)
        seg = partition_phones_in_event(10000, ["M", "AH", "N", "D", "EY"], rng)
        self.assertEqual(sum(seg), 10000)
        self.assertEqual(len(seg), 5)


class TestPhoneTemplates(unittest.TestCase):
    def test_store_roundtrip(self) -> None:
        tpl = PhoneTemplate(
            phone="T",
            channel_rms=np.ones(8),
            spd_diag_delta=np.arange(8, dtype=np.float64),
            rate_scale=1.2,
            duration_weight=0.35,
            n_segments=10,
        )
        store = PhoneTemplateStore(phones={"T": tpl}, n_channels=8, feature_dim=8, meta={})
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "tpl.json"
            save_phone_templates(store, path)
            from openalterego.sim.phonology.templates import load_phone_templates

            loaded = load_phone_templates(path)
        self.assertIn("T", loaded.phones)
        prof = loaded.channel_profile("T", 8)
        self.assertAlmostEqual(float(prof.sum()), 1.0, places=5)


class TestFitPhoneTemplates(unittest.TestCase):
    def test_fit_on_gowda_if_present(self) -> None:
        session = Path(__file__).resolve().parents[1] / "sessions" / "gowda_sv_full"
        if not (session / "signals.npy").is_file():
            self.skipTest("gowda_sv_full not present")
        store = fit_phone_templates(session, split="train", max_segments_per_phone=30, seed=7)
        self.assertGreater(len(store.phones), 10)
        self.assertGreater(store.phones["T"].n_segments, 0)


if __name__ == "__main__":
    unittest.main()
