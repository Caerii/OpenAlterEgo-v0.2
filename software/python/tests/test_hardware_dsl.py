"""Tests for hardware DSL (``.oae.json`` presets, validate, resolve, simulate)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from openalterego.hardware import (
    load_spec,
    resolve_all,
    resolve_sim_config,
    validate_spec,
)
from openalterego.hardware.load import list_presets
from openalterego.hardware.runner import run_chunk_simulation, run_virtual_ble_simulation_sync
from openalterego.hardware.validate import has_errors


REPO_ROOT = Path(__file__).resolve().parents[3]
SPECS_DIR = REPO_ROOT / "hardware" / "specs"


class TestHardwarePresets(unittest.TestCase):
    def test_all_presets_load(self) -> None:
        for name in list_presets():
            spec = load_spec(name)
            self.assertEqual(spec.name, name)

    def test_v0_openbci_validate_no_errors(self) -> None:
        issues = validate_spec(load_spec("v0_openbci"))
        self.assertFalse(has_errors(issues))

    def test_wide_fs_warning_on_250(self) -> None:
        spec = load_spec("v0_openbci")
        spec = spec.model_copy(
            update={"preprocess": spec.preprocess.model_copy(update={"mode": "wide"})}
        )
        codes = [i.code for i in validate_spec(spec)]
        self.assertIn("preprocess.wide_fs_low", codes)

    def test_resolve_emg_paradigm_standard(self) -> None:
        r = resolve_all(load_spec("v0_openbci"))
        self.assertEqual(r.emg_paradigm, "alterego_envelope")
        self.assertEqual(r.preprocess_mode, "standard")

    def test_resolve_v1_wide(self) -> None:
        r = resolve_all(load_spec("v1_wearable_ble"))
        self.assertEqual(r.emg_paradigm, "semg_literature_clamped")
        self.assertEqual(r.preprocess_mode, "wide")
        self.assertEqual(r.sim_config.fs_hz, 500)

    def test_sim_config_channels_match(self) -> None:
        cfg = resolve_sim_config(load_spec("tang_2025_headphone"))
        self.assertEqual(cfg.channels, 4)
        self.assertEqual(cfg.fs_hz, 1000)


class TestHardwareSpecFiles(unittest.TestCase):
    def test_repo_spec_files_parse(self) -> None:
        if not SPECS_DIR.is_dir():
            self.skipTest("hardware/specs not in checkout")
        for path in SPECS_DIR.glob("*.oae.json"):
            spec = load_spec(path)
            self.assertTrue(spec.name)

    def test_extends_merge(self) -> None:
        path = SPECS_DIR / "custom_wide_lab.oae.json"
        if not path.is_file():
            self.skipTest("custom_wide_lab spec missing")
        spec = load_spec(path)
        self.assertEqual(spec.sim.noise_uV, 40.0)
        self.assertEqual(spec.link.loss_prob, 0.03)
        self.assertEqual(spec.afe.fs_hz, 500)


class TestHardwareSimulate(unittest.TestCase):
    def test_chunk_simulation_runs(self) -> None:
        spec = load_spec("v0_openbci")
        chunks, rep = run_chunk_simulation(spec, duration_s=0.5)
        self.assertGreater(len(chunks), 0)
        self.assertGreater(rep.n_samples, 0)

    def test_virtual_ble_simulation_runs(self) -> None:
        spec = load_spec("v1_wearable_ble")
        rep = run_virtual_ble_simulation_sync(spec, duration_s=0.4)
        self.assertGreater(rep.packets_parsed, 0)
        self.assertGreater(rep.n_samples, 0)


class TestHardwareMontage(unittest.TestCase):
    def test_unknown_montage_fails(self) -> None:
        from openalterego.hardware.schema import HardwareSpec

        with self.assertRaises(ValueError):
            HardwareSpec.model_validate(
                {
                    "schema_version": 1,
                    "name": "bad",
                    "tier": "v0",
                    "electrodes": {"montage": "not_a_montage"},
                }
            )


class TestHardwareExportRoundtrip(unittest.TestCase):
    def test_json_roundtrip(self) -> None:
        from openalterego.hardware.schema import HardwareSpec

        raw = load_spec("wang_2021_tattoo").model_dump_public()
        blob = json.dumps(raw)
        again = HardwareSpec.model_validate(json.loads(blob))
        self.assertEqual(again.name, "wang_2021_tattoo")
        self.assertEqual(again.afe.fs_hz, 500)


if __name__ == "__main__":
    unittest.main()
