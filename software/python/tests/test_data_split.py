"""Stratified event splits for training/calibration."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from openalterego.ml.data_split import (
    gowda_sentence_train_val_indices,
    gowda_sim_transfer_merged_indices,
    resolve_gowda_train_val_test_indices,
    stratified_train_val_indices,
)


class TestStratifiedSplit(unittest.TestCase):
    def test_each_class_in_val_with_enough_rows(self) -> None:
        labels = np.array(["a"] * 20 + ["b"] * 20 + ["c"] * 20)
        tr, va = stratified_train_val_indices(labels, 0.25, seed=0)
        self.assertGreater(len(va), 0)
        va_l = set(labels[va])
        self.assertTrue({"a", "b", "c"}.issubset(va_l))

    def test_val_fraction_zero_all_train(self) -> None:
        labels = np.array(["x", "y", "z"])
        tr, va = stratified_train_val_indices(labels, 0.0, seed=1)
        self.assertEqual(len(va), 0)
        self.assertEqual(set(tr.tolist()), {0, 1, 2})

    def test_single_sample_class_goes_train(self) -> None:
        labels = np.array(["rare", "common", "common", "common", "common"])
        tr, va = stratified_train_val_indices(labels, 0.3, seed=2)
        rare_idx = int(np.flatnonzero(labels == "rare")[0])
        self.assertIn(rare_idx, tr)
        self.assertNotIn(rare_idx, va)

    def test_matches_event_dataframe_rows(self) -> None:
        df = pd.DataFrame(
            {"start_sample": range(30), "end_sample": range(1, 31), "label": ["p"] * 15 + ["q"] * 15}
        )
        tr, va = stratified_train_val_indices(df["label"].values, 0.2, seed=3)
        self.assertEqual(len(tr) + len(va), len(df))


class TestGowdaSentenceSplit(unittest.TestCase):
    def test_370_30_sentence_split(self) -> None:
        trial_ids = np.repeat(np.arange(400), 4)
        tr, va = gowda_sentence_train_val_indices(trial_ids, n_train_sentences=370, n_val_sentences=30)
        self.assertEqual(len(tr), 370 * 4)
        self.assertEqual(len(va), 30 * 4)
        self.assertEqual(set(trial_ids[tr].tolist()), set(range(370)))
        self.assertEqual(set(trial_ids[va].tolist()), set(range(370, 400)))


class TestSimTransferMergedSplit(unittest.TestCase):
    def test_real_block_goes_to_train(self) -> None:
        sim = np.repeat(np.arange(500), 4)
        real = np.repeat(np.arange(500, 537), 4)
        trial_ids = np.concatenate([sim, real])
        tr, va, te = gowda_sim_transfer_merged_indices(trial_ids)
        self.assertTrue(np.all(trial_ids[tr] >= 500) or np.any(trial_ids[tr] < 370))
        self.assertTrue(np.all((trial_ids[va] >= 370) & (trial_ids[va] < 400)))
        self.assertEqual(len(te), 100 * 4)
        self.assertGreater(len(tr), 370 * 4)

    def test_resolve_auto_merged_when_high_trial_id(self) -> None:
        trial_ids = np.array([0, 1, 500, 501])
        tr, va, te = resolve_gowda_train_val_test_indices(trial_ids)
        self.assertIn(2, tr.tolist())
        self.assertIn(3, tr.tolist())


if __name__ == "__main__":
    unittest.main()
