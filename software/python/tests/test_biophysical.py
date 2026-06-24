"""Biophysical-style MUAP + Poisson stream (MVP)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from openalterego.sim.biophysical.conduction import channel_delays_from_pickup
from openalterego.sim.biophysical.forward_model import green_pickup_matrix, motor_unit_pickup_weights
from openalterego.sim.biophysical.motor_pool import (
    firing_rates_baseline_segment,
    firing_rates_token_segment,
    init_motor_unit_layer,
)
from openalterego.sim.biophysical.muap import bipolar_muap_template, stretch_muap_template
from openalterego.sim.biophysical.pool import add_muap_spikes
from openalterego.sim.biophysical.recruitment import recruitment_rate_multipliers
from openalterego.sim.biophysical.stream import BiophysicalSimStream, BiophysicalSimStreamConfig
from openalterego.sim.biophysical.versions import BIOPHYS_MODEL_VERSION
from openalterego.sim.dataset import DatasetConfig, generate_dataset
from openalterego.sim.stream import ScenarioConfig, SimStreamConfig


class TestMuapTemplate(unittest.TestCase):
    def test_template_normalized_peak(self) -> None:
        m = bipolar_muap_template(1000.0, duration_ms=10.0)
        self.assertGreater(len(m), 8)
        self.assertLessEqual(float(np.max(np.abs(m))), 1.01)

    def test_stretch_widens_support(self) -> None:
        m0 = bipolar_muap_template(500.0, duration_ms=12.0)
        m1 = stretch_muap_template(m0, width_scale=1.65)
        self.assertGreater(m1.size, m0.size)
        self.assertLessEqual(float(np.max(np.abs(m1))), 1.01)


class TestMuapPool(unittest.TestCase):
    def test_add_spikes_finite(self) -> None:
        rng = np.random.default_rng(0)
        x = np.zeros((200, 3), dtype=np.float32)
        m = bipolar_muap_template(1000.0)
        add_muap_spikes(x, 1000.0, rng, 80.0, np.ones(3), m, 40.0)
        self.assertTrue(np.isfinite(x).all())
        self.assertGreater(float(np.std(x)), 0.0)

    def test_jitter_runs(self) -> None:
        rng = np.random.default_rng(1)
        x = np.zeros((500, 2), dtype=np.float32)
        m = bipolar_muap_template(1000.0)
        add_muap_spikes(
            x, 1000.0, rng, 120.0, np.ones(2), m, 30.0, time_jitter_std_s=0.002
        )
        self.assertTrue(np.isfinite(x).all())

    def test_spread_increases_cross_channel_correlation(self) -> None:
        rng_s = np.random.default_rng(42)
        rng_o = np.random.default_rng(42)
        n, c = 800, 4
        w = np.array([0.92, 0.04, 0.02, 0.02], dtype=np.float32)
        m = bipolar_muap_template(1000.0)
        xs = np.zeros((n, c), dtype=np.float32)
        xo = np.zeros((n, c), dtype=np.float32)
        add_muap_spikes(xs, 1000.0, rng_s, 250.0, w, m, 45.0, spread_across_channels=True)
        add_muap_spikes(xo, 1000.0, rng_o, 250.0, w, m, 45.0, spread_across_channels=False)

        def mean_pair_corr(z: np.ndarray) -> float:
            pairs: list[float] = []
            for i in range(c):
                for j in range(i + 1, c):
                    a, b = z[:, i].astype(np.float64), z[:, j].astype(np.float64)
                    if float(np.std(a)) < 1e-6 or float(np.std(b)) < 1e-6:
                        continue
                    pairs.append(float(np.corrcoef(a, b)[0, 1]))
            return float(np.mean(pairs)) if pairs else 0.0

        self.assertGreater(mean_pair_corr(xs), mean_pair_corr(xo) + 0.05)

    def test_delays_finite(self) -> None:
        rng = np.random.default_rng(0)
        w = np.array([0.5, 0.4, 0.08, 0.02], dtype=np.float32)
        d = channel_delays_from_pickup(w, 250.0, rng, velocity_m_s=4.0, jitter_samples=1)
        self.assertEqual(d.shape, (4,))
        self.assertTrue((d >= 0).all())


class TestForwardPickup(unittest.TestCase):
    def test_green_columns_normalize(self) -> None:
        G = green_pickup_matrix(6, 11, falloff=0.15)
        self.assertEqual(G.shape, (6, 11))
        self.assertTrue((G >= 0).all())
        np.testing.assert_allclose(np.sum(G, axis=0), 1.0, rtol=1e-5)

    def test_motor_weights_rows_normalize(self) -> None:
        rng = np.random.default_rng(1)
        G = green_pickup_matrix(4, 9, falloff=0.2)
        _, w = motor_unit_pickup_weights(rng, G, 20)
        self.assertEqual(w.shape, (20, 4))
        np.testing.assert_allclose(np.sum(w, axis=1), 1.0, rtol=1e-5)


class TestRecruitment(unittest.TestCase):
    def test_low_activation_fewer_large_units(self) -> None:
        g = np.array([0.5, 0.6, 1.2, 2.0, 2.5], dtype=np.float32)
        m_lo = recruitment_rate_multipliers(g, 0.15, steepness=12.0)
        m_hi = recruitment_rate_multipliers(g, 0.92, steepness=12.0)
        self.assertLess(float(np.sum(m_lo)), float(np.sum(m_hi)))


class TestMotorPoolRates(unittest.TestCase):
    def test_baseline_total_rate(self) -> None:
        rng = np.random.default_rng(0)
        _, _, g = init_motor_unit_layer(rng, 32, 4, 3)
        b = 18.0
        r = firing_rates_baseline_segment(g, b)
        self.assertAlmostEqual(float(np.sum(r)), b, places=5)

    def test_token_active_share(self) -> None:
        rng = np.random.default_rng(1)
        lab, _, g = init_motor_unit_layer(rng, 40, 4, 2)
        active_id = 0
        token_hz = 90.0
        base_hz = 12.0
        off_scale = 0.2
        r = firing_rates_token_segment(
            lab, g, active_label_id=active_id, token_firing_rate_hz=token_hz,
            baseline_firing_rate_hz=base_hz, off_label_rate_scale=off_scale,
        )
        m = lab == active_id
        self.assertAlmostEqual(float(np.sum(r[m])), token_hz, places=5)
        n_off = int(np.sum(~m))
        if n_off > 0:
            s_off = float(np.sum(g[~m].astype(np.float64))) + 1e-12
            expected_off = base_hz * off_scale * g[~m].astype(np.float64) / s_off
            np.testing.assert_allclose(r[~m], expected_off, rtol=1e-5)


class TestBiophysicalStream(unittest.TestCase):
    def test_generates_events_and_meta(self) -> None:
        cfg = BiophysicalSimStreamConfig(
            fs_hz=250,
            channels=4,
            seed=2,
            realtime_clock=False,
            band_limit_output=False,
            line_noise_uV=0.0,
            ar1_innovation_scale=0.0,
            realism_preset="off",
            volume_mix_mode="neighbor",
            use_forward_pickup=False,
            use_conduction_delays=False,
            use_recruitment=False,
            scenario=ScenarioConfig(labels=["a", "b"], p_event=0.95, event_duration_s=(0.2, 0.25), gap_duration_s=(0.05, 0.08)),
        )
        sim = BiophysicalSimStream(cfg)
        for _ in range(80):
            ch = sim.next_chunk()
            self.assertEqual(ch.meta.get("sim_engine"), "biophysical")
            self.assertEqual(ch.meta.get("sim_biophysical_model"), BIOPHYS_MODEL_VERSION)
            self.assertTrue(ch.meta.get("sim_motor_unit_pool"))
            self.assertEqual(ch.meta.get("sim_n_motor_units"), 48)
            self.assertFalse(ch.meta.get("sim_forward_pickup"))
            self.assertFalse(ch.meta.get("sim_conduction_delays"))
            self.assertFalse(ch.meta.get("sim_recruitment"))
        self.assertGreater(len(sim.events), 0)

    def test_realistic_defaults_stay_finite(self) -> None:
        cfg = BiophysicalSimStreamConfig(
            fs_hz=250,
            channels=6,
            seed=11,
            realtime_clock=False,
            scenario=ScenarioConfig(
                labels=["yes", "no"],
                p_event=0.7,
                event_duration_s=(0.15, 0.22),
                gap_duration_s=(0.08, 0.1),
            ),
        )
        sim = BiophysicalSimStream(cfg)
        for _ in range(30):
            ch = sim.next_chunk()
            self.assertTrue(np.isfinite(ch.samples).all())
            self.assertEqual(ch.meta.get("sim_biophysical_model"), BIOPHYS_MODEL_VERSION)
            self.assertEqual(ch.meta.get("sim_realism_preset"), "wearable")

    def test_realism_preset_parameterizes_sensor_pipeline(self) -> None:
        from types import SimpleNamespace

        from openalterego.sim.biophysical.sensor_pipeline import SensorNoiseState
        from openalterego.sim.realism import realism_preset_params

        self.assertEqual(realism_preset_params("off").pink_phi, 0.0)
        self.assertGreater(realism_preset_params("wearable").pink_phi, 0.99)

        def sensor_only_chunk(preset: str) -> np.ndarray:
            rng = np.random.default_rng(42)
            n, c = 200, 3
            x = np.zeros((n, c), dtype=np.float32)
            shim = SimpleNamespace(
                electrode_noise_uV=18.0,
                drift_uV_per_s=0.0,
                ar1_phi=0.0,
                ar1_innovation_scale=0.0,
                line_noise_uV=0.0,
                mains_freq_hz=60.0,
                realism_preset=preset,
            )
            SensorNoiseState(c, rng).apply_chunk(
                x, shim, rng, chunk_start=0, fs=250.0, noise_scale=1.0
            )
            return x

        x0 = sensor_only_chunk("off")
        x1 = sensor_only_chunk("wearable")
        self.assertGreater(float(np.std(x1)), float(np.std(x0)))

    def test_phoneme_drive_chunk_meta(self) -> None:
        cfg = BiophysicalSimStreamConfig(
            fs_hz=250,
            channels=4,
            seed=8,
            realtime_clock=False,
            band_limit_output=False,
            line_noise_uV=0.0,
            ar1_innovation_scale=0.0,
            realism_preset="off",
            volume_mix_mode="neighbor",
            use_forward_pickup=False,
            use_conduction_delays=False,
            use_recruitment=False,
            scenario=ScenarioConfig(
                labels=["yes"],
                p_event=1.0,
                event_duration_s=(0.2, 0.22),
                gap_duration_s=(0.1, 0.12),
                drive_mode="phoneme",
            ),
        )
        sim = BiophysicalSimStream(cfg)
        ch = sim.next_chunk()
        self.assertEqual(ch.meta.get("sim_drive_mode"), "phoneme")
        for _ in range(40):
            sim.next_chunk()
        self.assertGreater(len(sim.phoneme_events), 0)

    def test_v4_forward_meta(self) -> None:
        cfg = BiophysicalSimStreamConfig(
            fs_hz=250,
            channels=4,
            seed=7,
            realtime_clock=False,
            band_limit_output=False,
            line_noise_uV=0.0,
            ar1_innovation_scale=0.0,
            realism_preset="off",
            volume_mix_mode="neighbor",
            scenario=ScenarioConfig(
                labels=["a", "b"],
                p_event=0.5,
                event_duration_s=(0.12, 0.15),
                gap_duration_s=(0.05, 0.08),
            ),
        )
        sim = BiophysicalSimStream(cfg)
        ch = sim.next_chunk()
        self.assertTrue(ch.meta.get("sim_forward_pickup"))
        self.assertTrue(ch.meta.get("sim_conduction_delays"))
        self.assertTrue(ch.meta.get("sim_recruitment"))

    def test_legacy_pool_no_motor_meta(self) -> None:
        cfg = BiophysicalSimStreamConfig(
            fs_hz=250,
            channels=4,
            seed=3,
            realtime_clock=False,
            realism_preset="off",
            use_motor_unit_pool=False,
            scenario=ScenarioConfig(labels=["a", "b"], p_event=0.95, event_duration_s=(0.2, 0.25), gap_duration_s=(0.05, 0.08)),
        )
        sim = BiophysicalSimStream(cfg)
        ch = sim.next_chunk()
        self.assertFalse(ch.meta.get("sim_motor_unit_pool"))
        self.assertEqual(ch.meta.get("sim_n_motor_units"), 0)
        self.assertEqual(ch.meta.get("sim_biophysical_model"), "openalterego_biophysical_emg_v1")


class TestDatasetBiophysical(unittest.TestCase):
    def test_generate_dataset_biophysical_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "sess"
            ds = DatasetConfig(
                out_dir=out,
                duration_s=3.0,
                sim_engine="biophysical",
            )
            p = generate_dataset(ds)
            meta = (p / "meta.json").read_text(encoding="utf-8")
            self.assertIn("biophysical", meta)
            sig = np.load(p / "signals.npy")
            self.assertEqual(sig.ndim, 2)
            self.assertTrue(np.isfinite(sig).all())

    def test_generate_dataset_phoneme_tier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "sess"
            sc = ScenarioConfig(
                labels=["yes", "no"],
                p_event=0.9,
                drive_mode="phoneme",
            )
            sim_cfg = SimStreamConfig(
                fs_hz=250,
                channels=4,
                seed=21,
                scenario=sc,
                realtime_clock=False,
            )
            bcfg = BiophysicalSimStreamConfig(
                fs_hz=250,
                channels=4,
                seed=21,
                scenario=sc,
                realtime_clock=False,
                band_limit_output=False,
                line_noise_uV=0.0,
                ar1_innovation_scale=0.0,
                realism_preset="off",
                volume_mix_mode="neighbor",
                use_forward_pickup=False,
                use_conduction_delays=False,
                use_recruitment=False,
            )
            ds = DatasetConfig(
                out_dir=out,
                duration_s=4.0,
                config=sim_cfg,
                sim_engine="biophysical",
                biophysical=bcfg,
            )
            p = generate_dataset(ds)
            ph = p / "phonemes.csv"
            self.assertTrue(ph.is_file())
            txt = ph.read_text(encoding="utf-8")
            self.assertIn("phone", txt)
            self.assertIn("Y", txt)
            meta = (p / "meta.json").read_text(encoding="utf-8")
            self.assertIn("phoneme", meta.lower())
            self.assertIn('"drive_mode": "phoneme"', meta)


if __name__ == "__main__":
    unittest.main()
