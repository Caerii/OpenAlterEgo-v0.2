"""Lexicon Viterbi word decode from CTC frame posteriors (efficient closed vocab)."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from ..phonology.gowda_lexicon import BLANK_ID, phonemes_to_ids


def _ctc_expand_label(label: List[int], blank_id: int = BLANK_ID) -> List[int]:
    """Insert blanks between labels and at ends for CTC alignment."""
    out = [blank_id]
    for p in label:
        out.extend([int(p), blank_id])
    return out


def ctc_target_log_score(log_probs: np.ndarray, label_ids: List[int], *, blank_id: int = BLANK_ID) -> float:
    """Log P(label | x) via CTC forward algorithm. ``log_probs``: ``(T, C)``."""
    if not label_ids:
        return float("-inf")
    expanded = _ctc_expand_label(label_ids, blank_id=blank_id)
    t_len, _ = log_probs.shape
    s_len = len(expanded)
    if t_len == 0 or s_len == 0:
        return float("-inf")

    neg_inf = -1e30
    alpha = np.full((t_len, s_len), neg_inf, dtype=np.float64)
    lp = np.asarray(log_probs, dtype=np.float64)

    alpha[0, 0] = lp[0, expanded[0]]
    if s_len > 1:
        alpha[0, 1] = lp[0, expanded[1]]

    for t in range(1, t_len):
        for s in range(s_len):
            emit = lp[t, expanded[s]]
            stay = alpha[t - 1, s]
            step = alpha[t - 1, s - 1] if s >= 1 else neg_inf
            skip = neg_inf
            if s >= 2 and expanded[s] != blank_id and expanded[s] != expanded[s - 2]:
                skip = alpha[t - 1, s - 2]
            alpha[t, s] = emit + max(stay, step, skip)

    end0 = alpha[t_len - 1, s_len - 1]
    end1 = alpha[t_len - 1, s_len - 2] if s_len >= 2 else neg_inf
    return float(max(end0, end1))


def lexicon_viterbi_topk(
    log_probs: np.ndarray,
    input_length: int,
    lexicon: Dict[str, List[str]],
    *,
    topk: int = 8,
) -> List[Tuple[str, float]]:
    """Top-K lexicon words by CTC forward score."""
    t_len = min(int(input_length), int(log_probs.shape[0]))
    lp = np.asarray(log_probs[:t_len], dtype=np.float64)
    scored: List[Tuple[str, float]] = []
    for word, phones in lexicon.items():
        ids = phonemes_to_ids(phones)
        if not ids:
            continue
        scored.append((word, ctc_target_log_score(lp, ids)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[: max(1, int(topk))]


def lexicon_viterbi_word_rerank(
    log_probs: np.ndarray,
    input_length: int,
    lexicon: Dict[str, List[str]],
    *,
    lm_scores: Dict[str, float],
    lm_weight: float = 1.0,
    topk: int = 12,
) -> Tuple[str, float]:
    """Top-K CTC candidates reranked with per-word LM scores."""
    cands = lexicon_viterbi_topk(log_probs, input_length, lexicon, topk=topk)
    if not cands:
        return "", float("-inf")
    best_w, best_s = cands[0][0], float("-inf")
    for word, ctc_s in cands:
        total = float(ctc_s) + float(lm_weight) * float(lm_scores.get(word, math.log(1e-6)))
        if total > best_s:
            best_s = total
            best_w = word
    return best_w, best_s


def lexicon_viterbi_word(
    log_probs: np.ndarray,
    input_length: int,
    lexicon: Dict[str, List[str]],
) -> Tuple[str, float]:
    """Pick lexicon word maximizing CTC forward score."""
    t_len = min(int(input_length), int(log_probs.shape[0]))
    lp = np.asarray(log_probs[:t_len], dtype=np.float64)
    best_word = ""
    best_score = float("-inf")
    for word, phones in lexicon.items():
        ids = phonemes_to_ids(phones)
        if not ids:
            continue
        score = ctc_target_log_score(lp, ids)
        if score > best_score:
            best_score = score
            best_word = word
    return best_word, best_score


def lexicon_viterbi_batch(
    logits: np.ndarray,
    input_lengths: np.ndarray,
    lexicon: Dict[str, List[str]],
) -> List[str]:
    """Batch lexicon Viterbi from ``(B,T,C)`` logits."""
    log_probs = logits - np.max(logits, axis=-1, keepdims=True)
    log_probs = log_probs - np.log(np.sum(np.exp(log_probs), axis=-1, keepdims=True))
    out: List[str] = []
    for b in range(log_probs.shape[0]):
        word, _ = lexicon_viterbi_word(log_probs[b], int(input_lengths[b]), lexicon)
        out.append(word)
    return out
