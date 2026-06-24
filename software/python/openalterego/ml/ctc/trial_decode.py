"""Trial-ordered CTC decode with slot priors + word LM reranking."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..phonology.gowda_lexicon import ids_to_phonemes
from .dataset import PhonemeCTCDataset, ctc_collate_raw, ctc_collate_spd
from .lexicon_viterbi import lexicon_viterbi_topk, lexicon_viterbi_word_rerank
from .metrics import phoneme_error_rate, word_error_rate
from .trial_lm import TrialLanguageModel, fit_trial_lm, tune_lm_weight
from .util import input_lengths, unpack_batch


def _log_probs_from_logits(logits: np.ndarray) -> np.ndarray:
    lp = logits - np.max(logits, axis=-1, keepdims=True)
    return lp - np.log(np.sum(np.exp(lp), axis=-1, keepdims=True))


def _greedy_phoneme_ids(log_probs: np.ndarray, t_len: int) -> List[int]:
    from ..phonology.gowda_lexicon import BLANK_ID

    ids: List[int] = []
    prev = int(BLANK_ID)
    for t in range(int(t_len)):
        p = int(np.argmax(log_probs[t]))
        if p != int(BLANK_ID) and p != prev:
            ids.append(p)
        prev = p
    return ids


def _edit_distance(a: List[int], b: List[int]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev = dp[0]
        dp[0] = i
        for j, cb in enumerate(b, 1):
            cur = dp[j]
            cost = 0 if ca == cb else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return int(dp[-1])


def decode_word_trial_lm(
    log_probs: np.ndarray,
    input_length: int,
    lexicon: Dict[str, List[str]],
    lm: TrialLanguageModel,
    *,
    word_idx: int,
    prev_words: List[str],
    lm_weight: float = 1.0,
    topk: int = 16,
    full_slot_max: int = 120,
    phoneme_rerank_weight: float = 0.0,
    phoneme_rerank_slots: Tuple[int, ...] = (2, 3),
) -> str:
    slot_lex = lm.slot_candidates(int(word_idx), lexicon)
    n_slot = len(slot_lex)
    use_full = n_slot > 0 and n_slot <= int(full_slot_max)

    k = n_slot if use_full else min(int(topk), n_slot) if n_slot else int(topk)
    cands = lexicon_viterbi_topk(log_probs, input_length, slot_lex, topk=max(1, k))
    if not cands:
        cands = lexicon_viterbi_topk(log_probs, input_length, lexicon, topk=topk)

    if int(word_idx) in phoneme_rerank_slots and float(phoneme_rerank_weight) > 0:
        greedy_ids = _greedy_phoneme_ids(log_probs, input_length)
        rescored: List[Tuple[str, float]] = []
        pool = cands if use_full else cands[: min(12, len(cands))]
        for word, ctc_s in pool:
            from ..phonology.gowda_lexicon import phonemes_to_ids

            tid = phonemes_to_ids(lexicon.get(word, []))
            dist = _edit_distance(greedy_ids, tid)
            norm = max(len(tid), 1)
            rescored.append((word, float(ctc_s) - float(phoneme_rerank_weight) * dist / norm))
        rescored.sort(key=lambda x: x[1], reverse=True)
        cands = rescored

    if float(lm_weight) <= 0.0:
        return cands[0][0]

    lm_scores = {w: lm.context_logp(w, prev_words, int(word_idx)) for w, _ in cands}
    word, _ = lexicon_viterbi_word_rerank(
        log_probs,
        input_length,
        {w: lexicon[w] for w, _ in cands},
        lm_scores=lm_scores,
        lm_weight=float(lm_weight),
        topk=len(cands),
    )
    return word


def _trial_groups(dataset: PhonemeCTCDataset) -> Dict[int, List[int]]:
    groups: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for idx in range(len(dataset)):
        groups[int(dataset.trial_ids[idx])].append((int(dataset.word_indices[idx]), idx))
    out: Dict[int, List[int]] = {}
    for tid, pairs in groups.items():
        pairs.sort(key=lambda x: x[0])
        out[int(tid)] = [i for _, i in pairs]
    return out


def _forward_cache(
    model: torch.nn.Module,
    dataset: PhonemeCTCDataset,
    device: torch.device,
    *,
    batch_size: int = 32,
) -> Dict[int, Tuple[np.ndarray, int]]:
    """Run encoder once; return idx -> (log_probs, input_length)."""
    feature_type = str(getattr(dataset, "feature_type", "raw"))
    collate_fn = ctc_collate_spd if feature_type == "spd" else ctc_collate_raw
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, collate_fn=collate_fn, num_workers=0)

    cache: Dict[int, Tuple[np.ndarray, int]] = {}
    offset = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x, _, _, x_lens = unpack_batch(batch)
            x = x.to(device)
            logits = model(x)
            lengths = input_lengths(logits, x_lens)
            lp_batch = _log_probs_from_logits(logits.detach().cpu().numpy())
            for i in range(x.size(0)):
                idx = offset + i
                cache[idx] = (lp_batch[i], int(lengths[i].item()))
            offset += x.size(0)
    return cache


def evaluate_ctc_trial_lm(
    model: torch.nn.Module,
    dataset: PhonemeCTCDataset,
    device: torch.device,
    lm: TrialLanguageModel,
    *,
    lm_weight: float = 1.0,
    topk: int = 16,
    batch_size: int = 32,
) -> Dict[str, Any]:
    cache = _forward_cache(model, dataset, device, batch_size=int(batch_size))

    groups = _trial_groups(dataset)
    hyp_w: List[str] = [""] * len(dataset)
    ref_w = list(dataset.word_labels)
    hyp_ph: List[List[str]] = []
    ref_ph: List[List[str]] = []

    for _tid in sorted(groups.keys()):
        prev: List[str] = []
        for idx in groups[_tid]:
            lp, tlen = cache[idx]
            pred = decode_word_trial_lm(
                lp,
                tlen,
                dataset.lexicon,
                lm,
                word_idx=int(dataset.word_indices[idx]),
                prev_words=prev,
                lm_weight=float(lm_weight),
                topk=int(topk),
            )
            hyp_w[idx] = pred
            prev.append(pred)

    for i in range(len(dataset)):
        hyp_ph.append(dataset.lexicon.get(hyp_w[i], []))
        ref_ph.append(ids_to_phonemes(dataset.phoneme_seqs[i]))

    per = phoneme_error_rate(hyp_ph, ref_ph)
    wer = word_error_rate(hyp_w, ref_w)
    acc = float(np.mean([h == r for h, r in zip(hyp_w, ref_w)])) if ref_w else 0.0
    return {
        "per": round(per, 4),
        "wer": round(wer, 4),
        "word_acc": round(acc, 4),
        "n": len(ref_w),
        "decode_mode": "trial_lm",
        "lm_weight": float(lm_weight),
        "topk": int(topk),
        "hyp_words": hyp_w,
        "ref_words": ref_w,
    }


def tune_trial_lm_weight(
    model: torch.nn.Module,
    val_dataset: PhonemeCTCDataset,
    device: torch.device,
    lm: TrialLanguageModel,
    *,
    weights: Optional[List[float]] = None,
    batch_size: int = 32,
    topk: int = 16,
) -> float:
    weights = list(weights or [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])

    cache = _forward_cache(model, val_dataset, device, batch_size=int(batch_size))

    groups = _trial_groups(val_dataset)
    trials = []
    for tid in sorted(groups.keys()):
        items = []
        for idx in groups[tid]:
            lp, tlen = cache[idx]
            items.append(
                {
                    "idx": idx,
                    "ref": val_dataset.word_labels[idx],
                    "log_probs": lp,
                    "input_length": tlen,
                    "word_idx": int(val_dataset.word_indices[idx]),
                }
            )
        trials.append({"trial_id": tid, "items": items})

    def score_fn(item, lexicon, trial_lm, w, prev):
        return decode_word_trial_lm(
            item["log_probs"],
            item["input_length"],
            lexicon,
            trial_lm,
            word_idx=int(item["word_idx"]),
            prev_words=prev,
            lm_weight=float(w),
            topk=int(topk),
        )

    return tune_lm_weight(lm, lexicon=val_dataset.lexicon, score_fn=score_fn, trials=trials, weights=weights)
