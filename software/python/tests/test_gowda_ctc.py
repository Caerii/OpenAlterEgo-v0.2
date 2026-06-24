"""Tests for CTC metrics and phoneme lexicon."""

from __future__ import annotations

import unittest

from openalterego.ml.ctc.metrics import phoneme_error_rate, word_error_rate
from openalterego.ml.phonology.gowda_lexicon import build_lexicon, phonemes_to_ids, word_to_phonemes


class TestGowdaLexicon(unittest.TestCase):
    def test_monday_has_phonemes(self) -> None:
        ph = word_to_phonemes("monday")
        self.assertGreater(len(ph), 2)
        ids = phonemes_to_ids(ph)
        self.assertTrue(all(i > 0 for i in ids))

    def test_build_lexicon_weekdays(self) -> None:
        days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        lex = build_lexicon(days)
        self.assertEqual(len(lex), 7)


class TestCTCMetrics(unittest.TestCase):
    def test_per_identical(self) -> None:
        ref = [["M", "AH", "N"], ["T", "UW"]]
        self.assertEqual(phoneme_error_rate(ref, ref), 0.0)

    def test_wer_one_substitution(self) -> None:
        self.assertAlmostEqual(word_error_rate(["monday"], ["tuesday"]), 1.0)


if __name__ == "__main__":
    unittest.main()
