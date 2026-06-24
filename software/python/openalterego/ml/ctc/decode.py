"""CTC decode: greedy and prefix beam search (paper beam width 50)."""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from ..phonology.gowda_lexicon import BLANK_ID, PHONEME_ALPHABET, phonemes_to_ids


def greedy_ctc_decode(logits: torch.Tensor, input_lengths: torch.Tensor) -> List[List[int]]:
    """Decode ``(B, T, C)`` logits to phoneme id sequences (no blank repeats)."""
    if logits.dim() != 3:
        raise ValueError(f"expected (B,T,C), got {logits.shape}")
    pred = logits.argmax(dim=-1).cpu().numpy()
    lens = input_lengths.cpu().numpy().astype(int)
    out: List[List[int]] = []
    for b in range(pred.shape[0]):
        seq: List[int] = []
        prev = -1
        for t in range(int(lens[b])):
            p = int(pred[b, t])
            if p == BLANK_ID or p == prev:
                prev = p
                continue
            seq.append(p)
            prev = p
        out.append(seq)
    return out


def ids_to_phone_strings(batch_ids: List[List[int]]) -> List[List[str]]:
    inv = PHONEME_ALPHABET
    return [[inv[i] for i in seq if 0 < int(i) < len(inv)] for seq in batch_ids]


def _log_add(a: float, b: float) -> float:
    if a == float("-inf"):
        return b
    if b == float("-inf"):
        return a
    m = max(a, b)
    return m + float(np.log(np.exp(a - m) + np.exp(b - m)))


def beam_ctc_decode_logprobs(
    log_probs: np.ndarray,
    input_length: int,
    *,
    beam_width: int = 50,
    blank_id: int = BLANK_ID,
    allowed_prefixes: Optional[Set[Tuple[int, ...]]] = None,
) -> List[int]:
    """Prefix beam search on a single utterance. ``log_probs``: ``(T, C)``."""
    t_len = min(int(input_length), int(log_probs.shape[0]))
    lp = np.asarray(log_probs[:t_len], dtype=np.float64)
    n_classes = int(lp.shape[1])

    beams: Dict[Tuple[int, ...], List[float]] = {(): [0.0, float("-inf")]}

    for t in range(t_len):
        nxt: Dict[Tuple[int, ...], List[float]] = {}
        for prefix, (pb, pnb) in beams.items():
            for c in range(n_classes):
                p = float(lp[t, c])
                if c == blank_id:
                    key = prefix
                    if allowed_prefixes is not None and key not in allowed_prefixes:
                        continue
                    nb_pb, nb_pnb = nxt.get(key, [float("-inf"), float("-inf")])
                    nb_pb = _log_add(nb_pb, _log_add(pb + p, pnb + p))
                    nxt[key] = [nb_pb, nb_pnb]
                elif prefix and c == prefix[-1]:
                    key = prefix
                    if allowed_prefixes is None or key in allowed_prefixes:
                        nb_pb, nb_pnb = nxt.get(key, [float("-inf"), float("-inf")])
                        nb_pnb = _log_add(nb_pnb, pnb + p)
                        nxt[key] = [nb_pb, nb_pnb]
                    key2 = prefix + (c,)
                    if allowed_prefixes is None or key2 in allowed_prefixes:
                        nb_pb2, nb_pnb2 = nxt.get(key2, [float("-inf"), float("-inf")])
                        nb_pnb2 = _log_add(nb_pnb2, pb + p)
                        nxt[key2] = [nb_pb2, nb_pnb2]
                else:
                    key = prefix + (c,)
                    if allowed_prefixes is not None and key not in allowed_prefixes:
                        continue
                    nb_pb, nb_pnb = nxt.get(key, [float("-inf"), float("-inf")])
                    nb_pnb = _log_add(nb_pnb, _log_add(pb, pnb) + p)
                    nxt[key] = [nb_pb, nb_pnb]

        scored = sorted(nxt.items(), key=lambda kv: _log_add(kv[1][0], kv[1][1]), reverse=True)
        beams = dict(scored[: max(1, int(beam_width))])

    if not beams:
        return []
    best = max(beams.items(), key=lambda kv: _log_add(kv[1][0], kv[1][1]))[0]
    return list(best)


def build_lexicon_prefix_set(lexicon: Dict[str, List[str]]) -> Set[Tuple[int, ...]]:
    """All phoneme-id prefixes appearing in the session lexicon."""
    prefixes: Set[Tuple[int, ...]] = {()}
    for phones in lexicon.values():
        ids = phonemes_to_ids(phones)
        for i in range(len(ids) + 1):
            prefixes.add(tuple(ids[:i]))
    return prefixes


def beam_ctc_decode_batch(
    logits: torch.Tensor,
    input_lengths: torch.Tensor,
    *,
    beam_width: int = 50,
    lexicon_prefixes: Optional[Set[Tuple[int, ...]]] = None,
) -> List[List[int]]:
    """Beam decode a batch of ``(B, T, C)`` logits."""
    log_probs = torch.log_softmax(logits, dim=-1).detach().cpu().numpy()
    lens = input_lengths.detach().cpu().numpy().astype(int)
    out: List[List[int]] = []
    for b in range(log_probs.shape[0]):
        seq = beam_ctc_decode_logprobs(
            log_probs[b],
            int(lens[b]),
            beam_width=int(beam_width),
            allowed_prefixes=lexicon_prefixes,
        )
        out.append(seq)
    return out
