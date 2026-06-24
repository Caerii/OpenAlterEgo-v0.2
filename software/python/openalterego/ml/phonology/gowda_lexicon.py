"""Gowda small-vocab word → ARPABET phoneme sequences for CTC."""

from __future__ import annotations

import json
import re
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set

# Paper-style phoneme inventory (ARPABET, no stress digits)
PHONEME_ALPHABET: List[str] = [
    "<blank>",
    "AA", "AE", "AH", "AO", "AW", "AY",
    "B", "CH", "D", "DH", "EH", "ER", "EY",
    "F", "G", "HH", "IH", "IY", "JH", "K",
    "L", "M", "N", "NG", "OW", "OY", "P",
    "R", "S", "SH", "T", "TH", "UH", "UW",
    "V", "W", "Y", "Z", "ZH",
]
PHONEME_TO_ID: Dict[str, int] = {p: i for i, p in enumerate(PHONEME_ALPHABET)}
BLANK_ID = 0


def _normalize_word(word: str) -> str:
    s = re.sub(r"\s+", " ", str(word).strip().lower())
    return s.replace("_", "-")


def _strip_stress(phones: List[str]) -> List[str]:
    out: List[str] = []
    for p in phones:
        base = re.sub(r"\d", "", str(p).upper())
        if base in PHONEME_TO_ID:
            out.append(base)
    return out


@lru_cache(maxsize=1)
def _load_cmudict() -> Dict[str, List[str]]:
    """Load CMUdict pronunciations (cached)."""
    path = Path(__file__).with_name("cmudict_cache.json")
    if path.is_file():
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {k: v for k, v in raw.items()}

    url = "https://raw.githubusercontent.com/Alexir/CMUdict/master/cmudict-0.7b"
    text = urllib.request.urlopen(url, timeout=60).read().decode("utf-8", errors="ignore")
    out: Dict[str, List[str]] = {}
    for line in text.splitlines():
        if not line or line.startswith(";;;"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        word = re.sub(r"\(\d+\)$", "", parts[0]).lower()
        phones = _strip_stress(parts[1:])
        if phones and word not in out:
            out[word] = phones
    path.write_text(json.dumps(out), encoding="utf-8")
    return out


def _phones_for_token(token: str, cmu: Dict[str, List[str]]) -> List[str]:
    key = _normalize_word(token)
    for variant in (key, key.replace("-", " "), key.replace("-", "")):
        if variant in cmu:
            return list(cmu[variant])
    # hyphenated / multi-token years: "nineteen eighty" -> concat phones
    if " " in key or "-" in key:
        phones: List[str] = []
        for part in re.split(r"[\s\-]+", key):
            if not part:
                continue
            sub = _phones_for_token(part, cmu)
            if sub:
                phones.extend(sub)
        return phones
    # fallback: spell letter-by-letter using CMU entries for letters
    if len(key) <= 3 and key.isalpha():
        return _strip_stress(list(key.upper()))
    return []


def word_to_phonemes(word: str) -> List[str]:
    cmu = _load_cmudict()
    return _phones_for_token(word, cmu)


def build_lexicon(words: List[str]) -> Dict[str, List[str]]:
    lex: Dict[str, List[str]] = {}
    missing: List[str] = []
    for w in sorted(set(str(x) for x in words)):
        ph = word_to_phonemes(w)
        if ph:
            lex[w] = ph
        else:
            missing.append(w)
    if missing:
        raise ValueError(f"No phoneme mapping for words: {missing[:10]}")
    return lex


def phonemes_to_ids(phones: List[str]) -> List[int]:
    return [PHONEME_TO_ID[p] for p in phones if p in PHONEME_TO_ID]


def ids_to_phonemes(ids: List[int]) -> List[str]:
    inv = PHONEME_ALPHABET
    return [inv[i] for i in ids if 0 < int(i) < len(inv)]
