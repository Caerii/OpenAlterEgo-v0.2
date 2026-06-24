# Validation: Phase 2 (Ceiling, Bootstrap, CTC)

June 2026 — weekday ceiling, multi-seed bootstrap CI, and phoneme CTC path on `gowda_top30`.

**Command:**

```bash
openalterego analyze gowda-phase2 --data ./sessions/gowda_top30 --fs 5000 --device cuda
```

Sub-runs: `--only weekday`, `--only multiseed`, `--only ctc`

---

## 1. Weekday ceiling (7-way, slot 0)

Isolates **confusable weekday** tokens with winning preprocess (`gowda` + per-event + 2000 ms).

| Metric | Value |
|--------|-------|
| Classes | 7 (Mon–Sun) |
| Chance | 14.3% |
| **Val accuracy** | **50.0%** |
| Val macro-F1 | — |
| Train−val gap | small |

**Interpretation:** Weekdays are **much more learnable** than the full 30-way mix (36.8%). Remaining errors are mostly **across-day confusion** (expected). Full vocabulary dilutes accuracy with months/ordinals.

Checkpoint: `ablations/weekday7_ceiling.pt`

---

## 2. Multi-seed bootstrap (winning 30-way config)

Five seeds (`1337`–`1341`), per-event gowda 2000 ms, gowda split.

| Metric | Value |
|--------|-------|
| Val acc mean ± std | **27.9% ± ~4%** |
| **95% CI (bootstrap, pooled val)** | **[23.7%, 32.4%]** |
| Val macro-F1 mean | ~0.12 |

**Note:** Single-seed Phase 1 best was **36.8%**; seed variance is large on n=76 val events. Report bootstrap CI for honest uncertainty.

Checkpoints: `ablations/multiseed_seed*.pt`

---

## 3. Phoneme CTC (GRU baseline)

Paper-aligned **phoneme sequences** + CTC (not SPD σ(τ) yet). CMUdict lexicon, 40 ARPABET phones, 50 epochs.

| Metric | Train | Val |
|--------|-------|-----|
| **PER** | 36.9% | **48.6%** |
| **WER** (lexicon match) | 87.7% | **56.6%** |
| Word acc (lexicon) | 4.5% | **43.4%** |

Paper small-vocab target: PER **13%**, WER **14%**.

**Interpretation:**

- CTC + lexicon decode already reaches **~43% word match** on val without σ(τ) features — competitive with early CNN baseline (~20%) but **not** paper parity.
- Train WER >> val WER suggests **phoneme overfitting** / greedy decode; next steps: SPD features, beam search, HLG, more epochs.
- PER ~49% is far from paper 13% — architecture + features gap remains.

Checkpoint: `ablations/ctc_gowda.pt`

---

## Summary table

| Track | Key result | vs Phase 1 |
|-------|------------|------------|
| Weekday 7-way | **50% val** | Confirms vocab difficulty |
| 30-way + bootstrap | **~28% val (CI 24–32%)** | Honest uncertainty |
| CTC phoneme | **56.6% WER, 48.6% PER** | Paper path started |

---

## Code added (Phase 2)

| Module | Role |
|--------|------|
| `ml/eval/gowda_train.py` | Shared trainer + `eval_checkpoint` |
| `ml/eval/bootstrap.py` | Bootstrap CI |
| `ml/eval/gowda_phase2.py` | Phase 2 orchestrator |
| `ml/phonology/gowda_lexicon.py` | CMUdict → ARPABET |
| `ml/ctc/` | Dataset, GRU model, train, PER/WER |

---

## Recommended next steps (Phase 3)

1. **σ(τ) SPD features** + GRU (paper pipeline)
2. **Beam search** + 67-word lexicon for WER
3. **Full small-vocab import** (all 124 label strings) for fair paper comparison
4. **Multi-seed CTC** with bootstrap PER/WER CI

Raw JSON: `sessions/gowda_top30/ablations/phase2_report.json`
