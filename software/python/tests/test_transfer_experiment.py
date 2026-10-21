"""Experiment provenance and input-isolation checks."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.transfer_experiment import prepare


def _session(path: Path, trial_ids: list[int]) -> None:
    path.mkdir()
    (path / "signals.npy").write_bytes(b"fixture signal bytes")
    (path / "meta.json").write_text('{"fs_hz": 5000, "channels": 31}', encoding="utf-8")
    with (path / "events.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["trial_id", "label"])
        writer.writeheader()
        for trial_id in trial_ids:
            writer.writerow({"trial_id": trial_id, "label": "monday"})


class TestTransferExperiment(unittest.TestCase):
    def test_documented_499_trial_split_is_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sim, real = root / "sim", root / "real"
            _session(sim, [0])
            _session(real, list(range(499)))
            manifest, _ = prepare(sim, real, root / "run", device="cpu", seed=1)
            self.assertTrue(manifest["inputs"]["real"]["official_split_complete"])
            self.assertEqual(manifest["inputs"]["real"]["split"]["test_trials"], 99)

    def test_prepared_manifest_pins_inputs_and_keeps_outputs_separate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sim, real, out = root / "sim", root / "real", root / "run"
            _session(sim, [0, 370, 400])
            _session(real, [0, 370, 400])
            manifest, command = prepare(sim, real, out, device="cpu", seed=4)
            self.assertEqual(manifest["status"], "prepared")
            self.assertEqual(manifest["inputs"]["real"]["split"], {
                "train_trials": 1, "validation_trials": 1, "test_trials": 1,
            })
            self.assertEqual(len(manifest["inputs"]["sim"]["files"]["signals.npy"]["sha256"]), 64)
            self.assertIn(str(out / "work"), command)
            self.assertEqual(json.loads((out / "manifest.json").read_text())["parameters"]["seed"], 4)
            self.assertEqual(sorted(p.name for p in real.iterdir()), ["events.csv", "meta.json", "signals.npy"])
            with self.assertRaisesRegex(ValueError, "already exists"):
                prepare(sim, real, out, device="cpu", seed=4)

    def test_refuses_missing_held_out_split_and_output_inside_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sim, real = root / "sim", root / "real"
            _session(sim, [0])
            _session(real, [0, 370])
            with self.assertRaisesRegex(ValueError, "train, validation, and test"):
                prepare(sim, real, root / "run", device="cpu", seed=1)
            with self.assertRaisesRegex(ValueError, "outside both input"):
                prepare(sim, real, real / "run", device="cpu", seed=1)


if __name__ == "__main__":
    unittest.main()
