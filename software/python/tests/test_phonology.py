"""Phoneme lexicon and timeline helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from openalterego.sim.phonology import (
    expand_word_to_phones,
    iter_phone_slices,
    load_user_lexicon_overlay,
    merge_lexicon,
    partition_event_to_phones,
    phone_inventory,
    validate_lexicon,
)


class TestPhonology(unittest.TestCase):
    def test_expand_default_vocab(self) -> None:
        lex = merge_lexicon(None)
        self.assertEqual(expand_word_to_phones("yes", lex), ("Y", "EH", "S"))
        self.assertEqual(expand_word_to_phones("unknown", lex), ("@unknown",))

    def test_phone_inventory_stable(self) -> None:
        lex = merge_lexicon(None)
        inv = phone_inventory(["yes", "no"], lex)
        self.assertIn("Y", inv)
        self.assertIn("OW", inv)
        self.assertLess(len(inv), 20)

    def test_partition_sums(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(20):
            nt = int(rng.integers(8, 120))
            n = int(rng.integers(1, min(nt, 15)))
            parts = partition_event_to_phones(nt, n, rng)
            self.assertEqual(len(parts), n)
            self.assertEqual(sum(parts), nt)
            self.assertTrue(all(p >= 1 for p in parts))

    def test_iter_phone_slices_covers(self) -> None:
        seg = [10, 15, 20]
        chunks = iter_phone_slices(0, 45, seg)
        self.assertEqual(sum(b - a for a, b, _ in chunks), 45)
        chunks2 = iter_phone_slices(5, 20, seg)
        self.assertEqual(sum(b - a for a, b, _ in chunks2), 20)

    def test_validate_lexicon_empty_and_bad_token(self) -> None:
        self.assertEqual(validate_lexicon({"x": ("A",)}), [])
        self.assertTrue(any("empty" in m for m in validate_lexicon({"x": ()})))
        self.assertTrue(any("invalid" in m for m in validate_lexicon({"x": ("bad token!",)})))

    def test_merge_lexicon_uppercases_phones(self) -> None:
        m = merge_lexicon({"custom": ("aa", "bb")})
        self.assertEqual(m["custom"], ("AA", "BB"))

    def test_load_user_lexicon_overlay_file(self) -> None:
        data = {"zig": ["Z", "IH", "G"]}
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "lex.json"
            p.write_text(json.dumps(data), encoding="utf-8")
            user = load_user_lexicon_overlay(p)
        self.assertEqual(user["zig"], ("Z", "IH", "G"))
        merged = merge_lexicon(user)
        self.assertEqual(merged["zig"], ("Z", "IH", "G"))


if __name__ == "__main__":
    unittest.main()
