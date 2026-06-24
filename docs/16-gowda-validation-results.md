# Gowda Small-Vocab Validation Results (June 2026)

> **Canonical docs:** [`gowda/validation/01-small-vocab-subset.md`](gowda/validation/01-small-vocab-subset.md) · [`gowda/00-README.md`](gowda/00-README.md)  
> **Updated results (corrected import):** [`gowda/validation/02-top30-corrected.md`](gowda/validation/02-top30-corrected.md)

> **Update:** Full 499-trial / top-30-word results with **corrected import** are in [`gowda/validation/02-top30-corrected.md`](gowda/validation/02-top30-corrected.md). Early `gowda_sv` numbers below used a buggy importer (misaligned event indices + equal quarter splits).

Real closed-vocabulary validation using Gowda et al. **small-vocab** corpus from OSF [bgh7t](https://osf.io/bgh7t/).

## Data pipeline

| Step | Detail |
|------|--------|
| Download | `dataSmallVocab.npy` (~5.5 GB) + `labelsSmallVocab.npy` via OSF |
| Format | Data `(499, 31, 45000)` float64; labels `(1000, 4)` word grid per sentence template |
| Import | 200 trials → 12 top words → 300 word-level events (quarter splits per trial) |
| Session | `sessions/gowda_sv/` — fs=**5000 Hz**, **31 ch**, 675 s |

```bash
openalterego dataset import-gowda --download --out ./sessions/gowda_sv \
  --max-segments 200 --top-labels 12 --min-samples-per-label 5
```

## Preprocessing A/B (SE-ResNet, 20 epochs, 600 ms window)

| EMG mode | Val accuracy | Train accuracy |
|----------|--------------|----------------|
| standard | 10.9% | 33.9% |
| **wide** | 10.9% | **46.2%** |

12-way chance ≈ 8.3%. Wide bandpass fits training data better (literature-aligned).

## Full train (wide + SE-ResNet, 30 epochs)

```bash
openalterego train --data sessions/gowda_sv --fs 5000 --preprocess-mode streaming \
  --emg-mode wide --arch se_resnet --epochs 30 --segment-ms 600
```

Checkpoint: `sessions/gowda_sv/model.pt` — **best val accuracy 16.4%** (12-way, ~2× chance).

## Latency (600 ms window, 31 ch @ 5 kHz, CPU)

| Stage | p50 | p95 |
|-------|-----|-----|
| Preprocess | 1.2 ms | 1.6 ms |
| Inference window | 47.7 ms | **1497 ms** |
| Approx E2E p95 | — | **~1498 ms** |

31-channel @ 5 kHz is heavier than Gaddy (8 ch @ 1 kHz). p95 exceeds the 500 ms HCI target on CPU; GPU or smaller window/stride tuning recommended for deployment.

## Window sweep

| Window (ms) | Event acc | Latency p95 (ms) |
|-------------|-----------|------------------|
| 400 | 9.7% | 2007 |
| 600 | 9.3% | 276 |
| **900** | **10.0%** | **162** |
| 1200 | 10.3% | 335 |
| 1500 | 10.7% | 188 |

**Recommended window: 900 ms** (meets 500 ms p95 latency target on this hardware).

## Interpretation

1. **First real closed-vocab human orofacial EMG** through the full OpenAlterEgo stack (31 ch, 5 kHz, wide).
2. Val accuracy modest (~16% best during training) — expected without per-user calibration, single-subject word-quarter segmentation heuristic, and 12-class subset.
3. Still **above chance** on literature-aligned preprocessing, validating pipeline + model path.
4. Next: use full 499 trials, phoneme-aligned windows (paper timestamps), per-user train/val split, compare to Gowda reported GRU baselines.

## Reproduce

```bash
cd software/python
openalterego dataset import-gowda --download --out ./sessions/gowda_sv \
  --max-segments 200 --top-labels 12 --min-samples-per-label 5
openalterego dataset ab-preprocess --session ./sessions/gowda_sv --epochs 20
openalterego train --data ./sessions/gowda_sv --fs 5000 --emg-mode wide --arch se_resnet
```
