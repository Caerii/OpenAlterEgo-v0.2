"""Tests for per-user SPD basis store."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from openalterego.ml.spd.basis_store import fit_user_basis_from_session, load_user_basis, user_basis_exists


class TestBasisStore(unittest.TestCase):
    def test_fit_and_load_user_basis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp) / "sess"
            user = Path(tmp) / "user"
            session.mkdir()
            user.mkdir()
            n = 40000
            np.save(session / "signals.npy", np.random.randn(n, 8).astype(np.float32))
            pd.DataFrame(
                {
                    "start_sample": [0, 10000],
                    "end_sample": [8000, 18000],
                    "label": ["monday", "tuesday"],
                    "trial_id": [0, 1],
                    "word_idx": [0, 0],
                }
            ).to_csv(session / "events.csv", index=False)

            basis = fit_user_basis_from_session(session, user, max_matrices=20, seed=1)
            self.assertTrue(user_basis_exists(user))
            loaded = load_user_basis(user)
            np.testing.assert_allclose(loaded.basis_q, basis.basis_q, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
