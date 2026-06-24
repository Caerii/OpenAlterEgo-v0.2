"""Tests for CTC beam decode and test split."""

from __future__ import annotations

import unittest

import numpy as np

from openalterego.ml.ctc.decode import beam_ctc_decode_logprobs, build_lexicon_prefix_set, greedy_ctc_decode
from openalterego.ml.data_split import gowda_official_train_val_test_indices
from openalterego.ml.phonology.gowda_lexicon import BLANK_ID, PHONEME_ALPHABET, build_lexicon, phonemes_to_ids


class TestGowdaTestSplit(unittest.TestCase):
    def test_official_counts(self) -> None:
        trial_ids = np.repeat(np.arange(500), 4)
        tr, va, te = gowda_official_train_val_test_indices(trial_ids)
        self.assertEqual(len(tr), 370 * 4)
        self.assertEqual(len(va), 30 * 4)
        self.assertEqual(len(te), 100 * 4)


class TestBeamDecode(unittest.TestCase):
    def test_beam_finds_non_blank_path(self) -> None:
        # Two frames: prefer phoneme 1 then blank
        t, c = 4, len(PHONEME_ALPHABET)
        lp = np.full((t, c), -20.0, dtype=np.float64)
        pid = phonemes_to_ids(["M"])[0]
        lp[0, pid] = 0.0
        lp[1:, BLANK_ID] = 0.0
        seq = beam_ctc_decode_logprobs(lp, t, beam_width=5)
        self.assertEqual(seq, [pid])

    def test_lexicon_prefixes_include_empty(self) -> None:
        lex = build_lexicon(["monday", "tuesday"])
        prefs = build_lexicon_prefix_set(lex)
        self.assertIn((), prefs)
        self.assertGreater(len(prefs), 10)


if __name__ == "__main__":
    unittest.main()
