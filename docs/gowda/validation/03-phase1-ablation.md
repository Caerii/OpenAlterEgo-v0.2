# Validation: Phase 1 Preprocessing Ablation

June 2026 — systematic test of per-event DSP, paper bandpass, and window length on `gowda_top30`.

**Command:**

```bash
openalterego analyze gowda-ablation --data ./sessions/gowda_top30 --fs 5000 \
  --epochs 30 --batch-size 64 --device cuda
```

**Split:** gowda 370/30 sentences (908 train / 76 val) · **Model:** SE-ResNet · **AMP:** on (CUDA default)

---

## Results

| Config | Train acc | Val acc | Val macro-F1 | Train−val gap |
|--------|-----------|---------|--------------|---------------|
| baseline: streaming wide, 900 ms | 39.7% | 15.8% | 0.104 | **+23.9%** (overfit) |
| per-event wide, 900 ms | 26.1% | 19.7% | 0.089 | +6.4% |
| per-event **gowda** 80–1000 Hz, 900 ms | 31.9% | 14.5% | 0.066 | +17.5% |
| **per-event gowda, 2000 ms (full word)** | **39.7%** | **36.8%** | **0.187** | **+2.8%** |

Chance (30-way): **3.3%**. Previous best (streaming wide 900 ms): **19.7%**.

**Winner:** `per_event_gowda_2000` → checkpoint `sessions/gowda_top30/ablations/per_event_gowda_2000.pt`

---

## Scientific interpretation

### 1. Window length dominated

Word spans are **2 s (10 000 samples)**. A random **900 ms crop** discards most articulatory content and adds label noise. Using the **full 2 s window** (`--segment-ms 2000`) nearly **doubled** val accuracy.

### 2. Per-event preprocess removes streaming bleed

`--per-event-preprocess` applies 80–1000 Hz bandpass + z-score **inside each word span** with no cross-trial filter/EMA state. The baseline streaming path overfits (40% train / 16% val).

### 3. Paper bandpass helps only with full windows

Gowda 80–1000 Hz at 900 ms alone **underperformed** per-event wide at 900 ms. Combined with **2000 ms** windows it matches the best train fit and generalizes (gap ≈ 3%).

### 4. Still not paper WER 14%

This remains **30-way word classification** with a CNN — not phoneme CTC + SPD. **36.8% val acc** is a large step but PER/WER parity requires Phase E (phoneme labels + sequence decoder).

---

## Recommended training command (new default for Gowda)

```bash
openalterego train --data ./sessions/gowda_top30 --fs 5000 \
  --emg-mode gowda --per-event-preprocess \
  --segment-ms 2000 --split-by gowda --arch se_resnet \
  --epochs 30 --batch-size 64 --device cuda
```

---

## Artifacts

| File | Description |
|------|-------------|
| `sessions/gowda_top30/ablations/phase1_report.json` | Full metrics + per-class recall |
| `sessions/gowda_top30/ablations/phase1_run.log` | Console log |
| `sessions/gowda_top30/ablations/per_event_gowda_2000.pt` | Best checkpoint |

---

## Code added

| Module | Change |
|--------|--------|
| `dsp/filters.py` | `emg_mode=gowda` → 80–1000 Hz, 3rd-order Butterworth |
| `dsp/emg_config.py` | `validate_emg_gowda_fs()` |
| `ml/segment_cache.py` | `per_event_preprocess` in `build_segment_arrays` |
| `ml/train.py` | `--per-event-preprocess`, `--emg-mode gowda` |
| `ml/eval/gowda_ablation.py` | Phase 1 matrix runner |
| `ml/eval/metrics.py` | macro-F1 + per-class recall |
