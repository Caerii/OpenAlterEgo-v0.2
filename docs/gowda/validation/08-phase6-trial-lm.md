# Phase 6: Deep Error Analysis + Trial-Context Decode

Session: `software/python/sessions/gowda_sv_full`  
Checkpoint: `ablations/ctc_spd_v3_diag_delta_seed1337.pt`

## Command

```bash
uv run openalterego analyze gowda-phase6 --data ./sessions/gowda_sv_full --device cuda
```

Sections: baseline error taxonomy (`beam_lexicon`) + slot-aware trial decode (`trial_lm`).

## Error taxonomy (test, 396 words)

Baseline **beam_lexicon** — 28 errors (7.07% WER):

| Slot | Role | Errors | Error rate |
|------|------|--------|------------|
| 0 | weekday | 4 | 4.0% |
| 1 | month | 0 | 0% |
| 2 | ordinal | 9 | 9.0% |
| 3 | year | 15 | 15.0% |

| Category | Count | % of errors |
|----------|-------|-------------|
| year_same_era | 9 | 32% |
| ordinal | 9 | 32% |
| year_truncation | 6 | 21% |
| weekday | 4 | 14% |

**54% of remaining errors are year compounds** (same-era digit swap or truncation, e.g. `two_thousand_nine` → `two_thousand_five`).

Top confusions (baseline):

1. `two_thousand_nine` → `two_thousand_five` (3×)
2. `thirteenth` → `fourteenth` (2×)
3. `nineteen_seventy_eight` → `nineteen_seventy_nine` (1×)
4. `wednesday` → `may` (1×) — cross-slot; fixed by slot filter

## Phase 6 decode: slot-aware trial LM

Gowda trials are **4-word sentences** with fixed slots (weekday, month, ordinal, year). Phase 6:

1. Fits **slot priors** + bigram/trigram on train trials (`trial_lm.py`).
2. Decodes each trial **in order** with **slot-restricted lexicon Viterbi** (74 year labels, 31 ordinals, etc.).
3. Optional LM rerank (tuned on val); weight **0.0** — val already perfect without LM.

### Test results

| Decode | PER | WER | Word acc |
|--------|-----|-----|----------|
| v3 + beam_lexicon (Phase 5) | 3.2% | **7.07%** | 92.9% |
| v3 + **trial_lm** (Phase 6) | 2.8% | **6.82%** | **93.2%** |

Improvement is **decode-only** (~1 error recovered): slot filtering removes cross-slot mistakes (`wednesday` → `may`).

## What is *not* fixable by decoding

Per-error CTC forward scores show the model often **confidently prefers the wrong hypothesis** within the correct slot:

- `fourth` → `eighth`: CTC score −4.5 vs −9.1 for the correct word
- `two_thousand_nine` → `two_thousand_five`: wrong year wins by a large margin (3 cases)

Greedy-phoneme tie-breaking and LM weights (grid-searched 0–2.0 on val) did **not** change test WER — the bottleneck is now **acoustic discrimination** for similar compound words, especially years and ordinals sharing prefixes (`thirteen-*`, `nineteen_seventy_*`, `two_thousand_*`).

## Recommended next steps (training / data)

1. **Hard-negative fine-tuning** on confusable pairs from error analysis (year tails, ordinal teens).
2. **Longer σ context** or multi-scale SPD for compound words (4+ syllables).
3. **Multi-seed bootstrap** on v3 + trial_lm for confidence intervals.
4. **Sentence-level CTC** across 4-word trials (joint acoustic model; LM already encodes slot grammar).

## Modules

- `ml/ctc/trial_lm.py` — slot + n-gram LM from train trials
- `ml/ctc/trial_decode.py` — trial-ordered slot Viterbi decode
- `ml/eval/gowda_phase6.py` — error analysis + trial decode runner
