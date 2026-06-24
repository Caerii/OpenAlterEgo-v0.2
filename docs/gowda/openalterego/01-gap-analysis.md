# Gap Analysis: Paper vs OpenAlterEgo

Reference: [arXiv:2502.05762v2](https://arxiv.org/html/2502.05762v2)

---

## Side-by-side

| Dimension | Gowda 2025 (small-vocab) | OpenAlterEgo (gowda_sv_full, SPD v3) |
|-----------|--------------------------|--------------------------------------|
| **Goal** | Silent EMG → phoneme sequence → words | Same |
| **Features** | σ(τ) SPD 31×31 per 50 ms step | **diag+delta σ** (62-d) + cache |
| **Model** | 3-layer GRU + CTC | 2-layer **Bi**GRU + LayerNorm + dropout |
| **Metric** | PER / WER on **test** | Same split (370/30/100 sentences) |
| **Best test (paper-like)** | PER ~13%, WER ~14% | **PER 2.8%, WER 6.8%** (trial_lm) |
| **Best test (closed vocab)** | Lexicon match | **WER 7.1%** (beam_lexicon); **6.8%** (slot trial decode) |

---

## What we validate today

✅ Full small-vocab import (124 labels, 1996 word events)  
✅ SPD σ(τ) + CTC on official sentence splits  
✅ **Test-set PER/WER at or better than paper** (SPD v2)  
✅ Beam-50 decode + lexicon-constrained extension  
✅ Reproducible Phase 1–6 CLI (`gowda-ablation` … `gowda-phase6`)  
✅ **~2× better than paper WER** on honest test split (SPD v3 + decode)

---

## Remaining gaps

❌ Multi-seed bootstrap CI on test (planned: `gowda-phase4 --only multiseed`)  
❌ Large-vocab sentence-level CTC + HLG  
❌ Reference-channel subtraction (not in OSF 31-ch export)  
✅ **Sim→real transfer harness** — v1 scoreboard [`validation/09-sim-transfer.md`](validation/09-sim-transfer.md); v2 uses real SPD basis + anchor finetune

✅ Gowda-shaped biophysical sim (`sim-dataset --scenario gowda_sv`)  
✅ Offline trial decode (`decode-utterance`)  
✅ `analyze sim-transfer` harness (train sim → test real)

---

## Result timeline

| Phase | Task | Test PER / WER (best) |
|-------|------|------------------------|
| 1 | Word CNN + preprocess | N/A (classification) |
| 2 | Raw CNN+CTC | Not evaluated on test |
| 3 | SPD v1 + greedy val | Val only: 9.8% / 11.7% |
| 3+ | SPD v1 + test decode | 12.7% / 19.4% (beam) |
| **4** | **SPD v2 + test** | **10.1% / 14.1%** (beam); **10.4% WER** (lexicon-beam) |
| **5** | **SPD v3 diag_delta** | **3.2% / 7.1%** (beam_lexicon) |
| **6** | **v3 + slot trial decode** | **2.8% / 6.8%** (error analysis in `08-phase6-trial-lm.md`) |

The dominant lesson: **SPD geometry + CTC + closed-vocab decode + honest test eval** beat the paper; remaining errors are **compound-word acoustics** (years/ordinals), not CNN capacity or preprocessing.
