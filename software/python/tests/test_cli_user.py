"""CLI user / EMG config smoke tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from openalterego.cli import _cmd_user
from openalterego.dsp.emg_config import build_online_preprocessor


class TestCliUser(unittest.TestCase):
    def test_user_create_list_show_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ud = Path(tmp) / "userdata"
            ud.mkdir()
            self.assertEqual(_cmd_user(["--users-dir", str(ud), "create", "--user-id", "alice"]), 0)
            self.assertEqual(_cmd_user(["--users-dir", str(ud), "create", "--user-id", "alice"]), 1)
            self.assertEqual(_cmd_user(["--users-dir", str(ud), "list"]), 0)
            self.assertEqual(_cmd_user(["--users-dir", str(ud), "show", "--user-id", "alice", "--json"]), 0)
            self.assertEqual(_cmd_user(["--users-dir", str(ud), "delete", "--user-id", "alice", "-y"]), 0)


class TestEmgConfig(unittest.TestCase):
    def test_build_online_preprocessor_fallback_wide_low_fs(self) -> None:
        # wide invalid at 250 Hz -> falls back to standard inside build_online_preprocessor
        p = build_online_preprocessor(fs_hz=250.0, channels=4, emg_mode="wide")
        self.assertEqual(p.fs_hz, 250.0)


if __name__ == "__main__":
    unittest.main()
