"""CTC metrics: PER and lexicon WER."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple


def _levenshtein(a: Sequence[str], b: Sequence[str]) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return int(prev[m])


def phoneme_error_rate(hyp: List[List[str]], ref: List[List[str]]) -> float:
    dist = sum(_levenshtein(h, r) for h, r in zip(hyp, ref))
    n = sum(len(r) for r in ref)
    return float(dist / max(n, 1))


def word_error_rate(hyp_words: List[str], ref_words: List[str]) -> float:
    dist = _levenshtein(hyp_words, ref_words)
    return float(dist / max(len(ref_words), 1))


def match_word_from_phonemes(
    phones: List[str],
    lexicon: Dict[str, List[str]],
) -> str:
    """Pick lexicon word with minimum phoneme edit distance."""
    if not phones:
        return ""
    best_word = ""
    best_dist = 10**9
    for word, ref in lexicon.items():
        d = _levenshtein(phones, ref)
        if d < best_dist:
            best_dist = d
            best_word = word
    return best_word


def word_from_phoneme_ids(
    ids: List[int],
    lexicon: Dict[str, List[str]],
    *,
    phonemes_to_ids_fn=None,
    ids_to_phonemes_fn=None,
) -> str:
    """Exact lexicon match on phoneme ids, else edit-distance fallback."""
    from ..phonology.gowda_lexicon import ids_to_phonemes, phonemes_to_ids

    p2i = phonemes_to_ids_fn or phonemes_to_ids
    i2p = ids_to_phonemes_fn or ids_to_phonemes
    target = tuple(int(i) for i in ids)
    for word, phones in lexicon.items():
        if tuple(p2i(phones)) == target:
            return word
    return match_word_from_phonemes(i2p(ids), lexicon)
