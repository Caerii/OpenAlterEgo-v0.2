"""Tests for lexicon Viterbi CTC word decode."""

from __future__ import annotations

import unittest

import numpy as np

from openalterego.ml.ctc.lexicon_viterbi import ctc_target_log_score, lexicon_viterbi_word
from openalterego.ml.phonology.gowda_lexicon import build_lexicon, phonemes_to_ids


class TestLexiconViterbi(unittest.TestCase):
    def test_prefers_matching_word(self) -> None:
        lex = build_lexicon(["monday", "tuesday"])
        ids_mon = phonemes_to_ids(lex["monday"])
        t, c = 20, 41
        lp = np.full((t, c), -8.0, dtype=np.float64)
        for i, pid in enumerate(ids_mon):
            frame = min(t - 1, i * 2 + 1)
            lp[frame, pid] = -0.1
        word, _ = lexicon_viterbi_word(lp, t, lex)
        self.assertEqual(word, "monday")
        self.assertGreater(ctc_target_log_score(lp, ids_mon), ctc_target_log_score(lp, phonemes_to_ids(lex["tuesday"])))


if __name__ == "__main__":
    unittest.main()
