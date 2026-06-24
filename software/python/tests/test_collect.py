"""Tests for data collection (sim path; BLE requires hardware)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from openalterego.cli import _cmd_collect
from openalterego.users.collect import collect_from_sim


class TestCollectSim(unittest.TestCase):
    def test_collect_from_sim_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "sess"
            collect_from_sim(
                output_dir=out,
                user_id="tuser",
                duration_s=20.0,
                fs_hz=250,
                channels=8,
                seed=7,
                labels=["yes", "no"],
                p_event=0.95,
                preprocessing_mode="standard",
            )
            self.assertTrue((out / "signals.npy").exists())
            self.assertTrue((out / "events.csv").exists())
            self.assertTrue((out / "session.json").exists())
            sig = np.load(out / "signals.npy")
            self.assertEqual(sig.ndim, 2)
            self.assertEqual(sig.shape[1], 8)
            ev = pd.read_csv(out / "events.csv")
            self.assertGreater(len(ev), 0)
            meta = json.loads((out / "session.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["user_id"], "tuser")
            self.assertEqual(meta["fs_hz"], 250)

    def test_cli_collect_sim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "s2"
            rc = _cmd_collect(
                [
                    "sim",
                    "--out",
                    str(out),
                    "--user-id",
                    "cli_u",
                    "--seconds",
                    "8",
                    "--p-event",
                    "0.9",
                    "--labels",
                    "a,b",
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue((out / "signals.npy").exists())


if __name__ == "__main__":
    unittest.main()
