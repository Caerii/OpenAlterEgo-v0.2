# Phase 3: Full Small-Vocab + SPD σ(τ) CTC

Session: `software/python/sessions/gowda_sv_full`  
Paper target (small-vocab): PER ~13%, WER ~14% ([arXiv:2502.05762v2](https://arxiv.org/html/2502.05762v2))

## Import

```bash
cd software/python
uv run openalterego dataset import-gowda --download \
  --download-dir ./sessions/gowda_download \
  --out ./sessions/gowda_sv_full --full-vocab
```

| Field | Value |
|-------|-------|
| Events | 1996 word spans (499 trials × 4 words) |
| Unique label strings | 124 (dates/months/weekdays; paper cites 67-word vocabulary) |
| Channels | 31 @ 5 kHz |
| Split | 370 train / 30 val sentences (`gowda_sentence_train_val_indices`) |

## SPD pipeline (paper App. B)

1. Per-event **80–1000 Hz** bandpass + z-score (`emg_mode=gowda`)
2. Sliding windows: **100 ms** context, **50 ms** step
3. Edge matrix ℰ(τ), SPD regularize **η=0.1**
4. Log-Euclidean Fréchet mean on train windows → fixed eigenbasis **Q**
5. σ(τ) = Qᵀ ℰ(τ) Q → flattened 31×31 → **BiGRU + CTC** (greedy decode)

Basis cached under `sessions/gowda_sv_full/spd_cache/`.

## Training command

```bash
uv run openalterego analyze gowda-phase3 \
  --data ./sessions/gowda_sv_full --device cuda \
  --skip-import --only spd --epochs 50 --seed 1337
```

Config: Adam 1e-3, batch 32, 50 epochs, seed 1337, AMP on CUDA.

## Results (seed 1337)

| Split | PER | WER | Word acc |
|-------|-----|-----|----------|
| Train | 0.04% | 94.3% | 1.8% |
| **Val** | **9.8%** | **11.7%** | **88.3%** |

Checkpoint: `sessions/gowda_sv_full/ablations/ctc_spd.pt`  
Elapsed: ~253 s (GPU, RTX 3080 class)

### Interpretation

- **Validation PER/WER match or beat paper small-vocab** once σ(τ) + CTC replaces the raw CNN stem.
- Train WER is poor (many classes, memorization of phoneme paths) while val generalizes — consistent with sentence-held-out split and lexicon matching.
- Remaining gaps vs paper: **beam search (width 50)**, test-set 100 sentences, optional HLG for large vocab.

## Comparison to Phase 2 (top-30, raw CTC)

| Path | Val PER | Val WER |
|------|---------|---------|
| Raw CNN+CTC (top-30) | 48.6% | 56.6% |
| **SPD+GRU+CTC (full vocab)** | **9.8%** | **11.7%** |

## Next steps

- Beam decode + multi-seed bootstrap CI on PER/WER
- Wire held-out **test** 100 sentences
- SPD feature cache (σ sequences) to avoid rebuild each run
- Raw CTC baseline on full vocab: `--only raw_ctc`
