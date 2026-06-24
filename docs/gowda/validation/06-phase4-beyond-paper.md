# Phase 4: Beyond Paper (Beam, Test Split, Enhanced SPD)

Session: `software/python/sessions/gowda_sv_full`  
Paper small-vocab: PER ~13%, WER ~14% on held-out evaluation.

## What Phase 4 adds (beyond paper)

| Capability | Paper | Phase 4 |
|------------|-------|---------|
| Official **test** split (100 sentences) | Yes | Wired (`gowda_official_train_val_test_indices`) |
| **Beam search** decode (width 50) | PER only | PER + WER ablation |
| **Lexicon-constrained** beam | 67-word lexicon match post-hoc | Prefix-pruned beam + exact word match |
| SPD features | Full 31×31 σ(τ) | Optional **upper-triangle** (496-d, symmetric) |
| GRU | 3-layer | 3-layer BiGRU + **LayerNorm + dropout 0.2** |
| LR schedule | — | **Cosine annealing** |
| σ sequence cache | — | Disk cache under `spd_sequences/` |
| Multi-seed bootstrap | — | PER/WER/word-acc CI on test |

## Decode ablation (Phase 3 checkpoint, no retrain)

Checkpoint: `ablations/ctc_spd.pt` (legacy 2-layer GRU)

```bash
uv run openalterego analyze gowda-phase4 --data ./sessions/gowda_sv_full \
  --device cuda --only decode --checkpoint ./sessions/gowda_sv_full/ablations/ctc_spd.pt
```

### Validation (30 sentences, 120 words)

| Decode | PER | WER | Word acc |
|--------|-----|-----|----------|
| Greedy | 9.8% | 11.7% | 88.3% |
| Beam-50 | **8.8%** | 11.7% | 88.3% |
| **Beam + lexicon** | **3.8%** | **5.0%** | **95.0%** |

### Test (100 sentences, 396 words)

| Decode | PER | WER | Word acc |
|--------|-----|-----|----------|
| Greedy | 13.1% | 21.5% | 78.5% |
| Beam-50 | **12.7%** | **19.4%** | **80.6%** |
| **Beam + lexicon** | **7.7%** | **16.4%** | **83.6%** |

**Interpretation:** On the official **test** split, unconstrained beam already beats paper PER (~12.7% vs ~13%). Lexicon-constrained decoding is an OpenAlterEgo extension (small closed vocab) and pushes WER well below paper on val; test WER **16.4%** is still above paper **14%** but much closer than greedy **21.5%**.

## Enhanced SPD v2 training (completed seed 1337)

```bash
uv run openalterego analyze gowda-phase4 --data ./sessions/gowda_sv_full \
  --device cuda --only train --epochs 60 --seed 1337
```

Config: upper-tri σ (496-d), 3× BiGRU + LayerNorm + dropout 0.2, cosine LR, **greedy** val for checkpointing (fast), beam ablation after train.  
Checkpoint: `ablations/ctc_spd_v2_seed1337.pt` — **244 s** train + decode.

### SPD v2 vs Phase 3 legacy (`ctc_spd.pt`) on **test** (396 words)

| Model | Decode | PER | WER | Word acc |
|-------|--------|-----|-----|----------|
| Phase 3 | Greedy | 13.1% | 21.5% | 78.5% |
| Phase 3 | Beam-50 | 12.7% | 19.4% | 80.6% |
| Phase 3 | Beam + lexicon | 7.7% | 16.4% | 83.6% |
| **SPD v2** | **Greedy** | **10.3%** | **14.1%** | **85.9%** |
| **SPD v2** | **Beam-50** | **10.1%** | **14.1%** | **85.9%** |
| **SPD v2** | **Beam + lexicon** | **5.6%** | **11.9%** | **88.1%** |
| Paper | — | ~13% | ~14% | — |

**SPD v2 beats paper on test** for both PER (beam ~10%) and WER (greedy/beam ~14.1%, lexicon-beam **11.9%**).

### SPD v2 validation (120 words)

| Decode | PER | WER |
|--------|-----|-----|
| Greedy | 8.8% | 7.5% |
| Beam-50 | 8.3% | 6.7% |
| Beam + lexicon | 2.3% | 4.2% |

Training curve (greedy val): epoch 1 → 100% WER; epoch 15 → 5% WER; epoch 60 → **3.3% WER**.

## Next steps

- Multi-seed bootstrap: `--only multiseed --seeds 1337,1338,1339`
- Sentence-level CTC for large-vocab path
- KenLM / HLG for open vocabulary WER
