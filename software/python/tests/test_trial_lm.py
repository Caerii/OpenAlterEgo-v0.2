"""Tests for trial LM + top-k lexicon rerank."""

from __future__ import annotations

import numpy as np

from openalterego.ml.ctc.lexicon_viterbi import lexicon_viterbi_topk, lexicon_viterbi_word_rerank
from openalterego.ml.ctc.trial_lm import fit_trial_lm


def test_lexicon_viterbi_topk_returns_sorted():
    lex = {"aa": ["AA"], "ab": ["AA", "BB"]}
    lp = np.full((4, 3), -3.0, dtype=np.float64)
    lp[:, 0] = -0.1
    lp[0, 1] = -0.01
    lp[1, 1] = -0.01
    lp[0, 2] = -0.02
    lp[1, 2] = -0.02
    top = lexicon_viterbi_topk(lp, 4, lex, topk=2)
    assert len(top) >= 1
    if len(top) == 2:
        assert top[0][1] >= top[1][1]


def test_trial_lm_slot_vocab():
    import pandas as pd

    rows = []
    for tid in range(2):
        rows.extend(
            [
                {"trial_id": tid, "word_idx": 0, "label": "monday"},
                {"trial_id": tid, "word_idx": 1, "label": "january"},
                {"trial_id": tid, "word_idx": 2, "label": "first"},
                {"trial_id": tid, "word_idx": 3, "label": "nineteen_ninety"},
            ]
        )
    ev = pd.DataFrame(rows)
    lm = fit_trial_lm(ev, max_trial_id=10)
    assert "monday" in lm.slot_vocab[0]
    assert "january" in lm.slot_vocab[1]
    slot_lex = lm.slot_candidates(0, {"monday": ["M"], "february": ["F"]})
    assert list(slot_lex.keys()) == ["monday"]


def test_lexicon_rerank_prefers_lm():
    lex = {"aa": ["AA"], "bb": ["BB"]}
    lp = np.full((3, 3), -2.0, dtype=np.float64)
    lp[:, 0] = -0.05
    lp[0, 1] = -0.02
    lp[1, 1] = -0.02
    lp[0, 2] = -0.5
    lp[1, 2] = -0.5
    word, _ = lexicon_viterbi_word_rerank(
        lp, 3, lex, lm_scores={"aa": -0.01, "bb": -10.0}, lm_weight=1.0, topk=2
    )
    assert word == "aa"
