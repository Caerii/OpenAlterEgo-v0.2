"""Trial-level word language model + slot priors (Gowda 4-word sentences)."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

import pandas as pd

WEEKDAYS = frozenset(
    {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}
)
MONTHS = frozenset(
    {
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
    }
)


@dataclass
class TrialLanguageModel:
    """Bigram + slot priors fit on training trials only."""

    slot_vocab: Dict[int, Set[str]] = field(default_factory=dict)
    slot_logp: Dict[int, Dict[str, float]] = field(default_factory=dict)
    bigram_logp: Dict[str, Dict[str, float]] = field(default_factory=dict)
    trigram_logp: Dict[str, Dict[str, float]] = field(default_factory=dict)
    words_per_trial: int = 4
    backoff: float = 0.1

    def slot_candidates(self, word_idx: int, lexicon: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Restrict decode candidates using training slot vocabulary."""
        allowed = self.slot_vocab.get(int(word_idx))
        if not allowed:
            return lexicon
        return {w: p for w, p in lexicon.items() if w in allowed}

    def context_logp(self, word: str, prev_words: Sequence[str], word_idx: int) -> float:
        w = str(word)
        score = float(self.slot_logp.get(int(word_idx), {}).get(w, math.log(self.backoff / max(len(self.slot_vocab.get(int(word_idx), {w})), 1))))
        if prev_words:
            p1 = str(prev_words[-1])
            score += float(self.bigram_logp.get(p1, {}).get(w, math.log(self.backoff)))
        if len(prev_words) >= 2:
            key = f"{prev_words[-2]}|{prev_words[-1]}"
            score += float(self.trigram_logp.get(key, {}).get(w, math.log(self.backoff)))
        return score


def _log_smooth(count: int, total: int, vocab: int) -> float:
    return math.log((count + 0.1) / (total + 0.1 * max(vocab, 1)))


def fit_trial_lm(events: pd.DataFrame, *, max_trial_id: int = 370) -> TrialLanguageModel:
    """Fit on sentence-train trials only (official 370)."""
    ev = events[events["trial_id"].astype(int) < int(max_trial_id)].copy()
    ev = ev.sort_values(["trial_id", "word_idx"])

    slot_counts: Dict[int, Counter[str]] = defaultdict(Counter)
    bigram_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    trigram_counts: Dict[str, Counter[str]] = defaultdict(Counter)

    for tid, grp in ev.groupby("trial_id"):
        words = [str(x) for x in grp.sort_values("word_idx")["label"].tolist()]
        for wi, w in enumerate(words):
            slot_counts[int(wi)][w] += 1
        for i in range(1, len(words)):
            bigram_counts[words[i - 1]][words[i]] += 1
        for i in range(2, len(words)):
            trigram_counts[f"{words[i-2]}|{words[i-1]}"][words[i]] += 1

    slot_vocab: Dict[int, Set[str]] = {}
    slot_logp: Dict[int, Dict[str, float]] = {}
    for wi, ctr in slot_counts.items():
        slot_vocab[int(wi)] = set(ctr.keys())
        total = int(sum(ctr.values()))
        vocab = len(ctr)
        slot_logp[int(wi)] = {w: _log_smooth(int(c), total, vocab) for w, c in ctr.items()}

    bigram_logp: Dict[str, Dict[str, float]] = {}
    for w1, ctr in bigram_counts.items():
        total = int(sum(ctr.values()))
        vocab = len(ctr)
        bigram_logp[w1] = {w2: _log_smooth(int(c), total, vocab) for w2, c in ctr.items()}

    trigram_logp: Dict[str, Dict[str, float]] = {}
    for key, ctr in trigram_counts.items():
        total = int(sum(ctr.values()))
        vocab = len(ctr)
        trigram_logp[key] = {w2: _log_smooth(int(c), total, vocab) for w2, c in ctr.items()}

    # Enrich slot 0/1 with known weekday/month sets when train coverage is thin
    if 0 in slot_vocab:
        slot_vocab[0] = set(slot_vocab[0]) | WEEKDAYS
    if 1 in slot_vocab:
        slot_vocab[1] = set(slot_vocab[1]) | MONTHS

    return TrialLanguageModel(
        slot_vocab=slot_vocab,
        slot_logp=slot_logp,
        bigram_logp=bigram_logp,
        trigram_logp=trigram_logp,
    )


def tune_lm_weight(
    lm: TrialLanguageModel,
    *,
    lexicon: Dict[str, List[str]],
    score_fn,
    trials: List[dict],
    weights: Optional[List[float]] = None,
) -> float:
    """Grid-search LM weight on val trials. ``score_fn(lp, lex, lm_w) -> word``."""
    weights = list(weights or [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
    best_w = 0.0
    best_acc = -1.0
    for w in weights:
        correct = 0
        total = 0
        for trial in trials:
            prev: List[str] = []
            for item in trial["items"]:
                pred = score_fn(item, lexicon, lm, float(w), prev)
                if pred == item["ref"]:
                    correct += 1
                total += 1
                prev.append(pred)
        acc = correct / max(total, 1)
        if acc > best_acc:
            best_acc = acc
            best_w = float(w)
    return best_w
