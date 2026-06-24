# Phase 5: Efficient SPD v3 + Lexicon Viterbi

Session: `software/python/sessions/gowda_sv_full`

## Hypothesis

Paper argues off-diagonals of σ(τ) are small after diagonalization → **log eigenvalues + temporal delta** (62-d) should match 496-d upper-tri accuracy with **8× smaller** GRU input, enabling larger batches and faster training. **Lexicon Viterbi** scores each vocabulary word via CTC forward algorithm (better than beam + edit distance).

## Command

```bash
uv run openalterego analyze gowda-phase5 --data ./sessions/gowda_sv_full \
  --device cuda --epochs 80 --seed 1337 --feature-mode diag_delta
```

Config: 62-d diag+delta σ, SpecAugment-style noise/masking, 2× BiGRU (192 hidden), batch 64, warmup+cosine LR, 140 s train (vs 244 s v2).

## Test results (396 words, official 100-sentence split)

| Model / decode | PER | WER | Word acc | Train time |
|----------------|-----|-----|----------|------------|
| Paper | ~13% | ~14% | — | — |
| SPD v2 + beam | 10.1% | 14.1% | 85.9% | 244 s |
| SPD v2 + beam_lexicon | 5.6% | 10.4% | 89.7% | — |
| SPD v2 + **lexicon_viterbi** (decode only) | 4.8% | **9.9%** | **90.2%** | 0 s |
| **SPD v3 + beam_lexicon** | **3.2%** | **7.1%** | **92.9%** | **140 s** |
| SPD v3 + greedy | 5.0% | 8.6% | 91.4% | — |

Checkpoint: `ablations/ctc_spd_v3_diag_delta_seed1337.pt`

## Takeaways

1. **Compact features work** — diag+delta beats full upper-tri on test WER while training ~1.7× faster.
2. **Decoder matters as much as encoder** — lexicon Viterbi on v2 alone dropped test WER 14.1% → 9.9% with no retrain.
3. **Combined v3 + lexicon beam** reaches **~7% WER / ~93% word accuracy** on held-out test — roughly **2× better than paper WER**.

## Modules

- `ml/spd/features.py` — `diag`, `diag_delta` modes, log eigenvalues
- `ml/ctc/augment.py` — σ sequence augmentation
- `ml/ctc/lexicon_viterbi.py` — CTC forward word scoring
- `ml/eval/gowda_phase5.py` — Phase 5 runner
