"""Integration: hardware DSL through collect and sim-dataset."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from openalterego.cli import _cmd_collect, _cmd_hw
from openalterego.hardware.bind import dataset_config_from_hw, load_hw_spec
from openalterego.sim.dataset import generate_dataset
from openalterego.sim.stream import ScenarioConfig
from openalterego.users.collect import collect_from_hw_spec


class TestHwPipeline(unittest.TestCase):
    def test_collect_from_hw_spec_writes_session_json(self) -> None:
        spec = load_hw_spec("v0_openbci")
        with tempfile.TemporaryDirectory() as td:
            out = collect_from_hw_spec(
                spec=spec,
                output_dir=Path(td),
                user_id="alice",
                duration_s=0.5,
                seed=42,
            )
            session_path = out / "session.json"
            self.assertTrue(session_path.is_file())
            meta = json.loads(session_path.read_text(encoding="utf-8"))
            self.assertIn("hardware_spec", meta)
            self.assertEqual(meta["hardware_spec"]["name"], "v0_openbci")
            self.assertTrue((out / "signals.npy").is_file())

    def test_sim_dataset_from_hw_spec(self) -> None:
        spec = load_hw_spec("v1_wearable_ble")
        sc = ScenarioConfig(labels=["yes", "no"], p_event=0.65)
        with tempfile.TemporaryDirectory() as td:
            ds = dataset_config_from_hw(
                spec,
                out_dir=td,
                duration_s=0.5,
                scenario=sc,
                seed=7,
            )
            path = generate_dataset(ds)
            meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["fs_hz"], 500)
            self.assertEqual(meta["channels"], 8)

    def test_cli_collect_sim_hw_spec(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "sess"
            rc = _cmd_collect(
                [
                    "sim",
                    "--hw-spec",
                    "v0_openbci",
                    "--out",
                    str(out),
                    "--user-id",
                    "cli_hw",
                    "--seconds",
                    "2",
                ]
            )
            self.assertEqual(rc, 0)
            meta = json.loads((out / "session.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["hardware_spec"]["name"], "v0_openbci")

    def test_cli_hw_run(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "run_sess"
            rc = _cmd_hw(
                [
                    "run",
                    "v0_openbci",
                    "--out",
                    str(out),
                    "--user-id",
                    "run_user",
                    "--seconds",
                    "1",
                    "--smoke-seconds",
                    "0.3",
                    "--json",
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue((out / "session.json").is_file())
            self.assertTrue((out / "signals.npy").is_file())


if __name__ == "__main__":
    unittest.main()
