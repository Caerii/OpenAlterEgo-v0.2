# Gaddy EMG Validation Results (June 2026)

Real-data smoke test using [Gaddy silent speech EMG](https://doi.org/10.5281/zenodo.4064409) imported into OpenAlterEgo session format.

## Data pipeline

| Step | Command / artifact |
|------|---------------------|
| Download | Zenodo `emg_data.tar.gz` (~3.7 GB) → `software/python/datasets/gaddy_download/` |
| Import | `openalterego dataset import-gaddy --raw .../raw --out sessions/gaddy_cv --max-samples 250 --top-labels 8 --min-samples-per-label 3` |
| Session | `sessions/gaddy_cv/` — 90 events, 8 labels, fs=1000 Hz, 8 ch, 292.8 s |
| Labels | `april`, `august`, `june`, `saturday`, `sunday`, `thursday`, `tuesday`, `wednesday` (first-word from date-reading prompts) |

## Latency (real Gaddy checkpoint, fs=1000, SE-ResNet)

```bash
openalterego latency-bench --model sessions/gaddy_cv/model.pt --window-ms 600
```

| Stage | p50 | p95 |
|-------|-----|-----|
| Preprocess | 1.1 ms | 1.7 ms |
| Inference | 12.2 ms | 42.0 ms |
| **Approx E2E** | — | **43.7 ms** |

Well within Tang/AlterEgo **500 ms** target.

## Window sweep (event accuracy on same session)

| window_ms | p95 latency | event accuracy |
|-----------|-------------|----------------|
| 400 | 21 ms | 21.1% |
| 600 | 30 ms | 20.0% |
| 900 | 57 ms | 18.9% |
| 1200 | 96 ms | 20.0% |

Recommended: **400 ms** (best accuracy/latency trade-off on this pseudo-label set).

## Preprocessing A/B (SE-ResNet, 25 epochs, segment 600 ms)

Saved: `sessions/gaddy_cv/ab_preprocess.json`

| EMG mode | Raw SNR* | Val accuracy | Train accuracy |
|----------|----------|--------------|----------------|
| standard | -8.6 dB | 0.0% | 44.4% |
| **wide** | -26.6 dB | **6.3%** | **92.1%** |

\*SNR computed on **unfiltered** raw stack; negative values reflect open-vocab recording conditions and band choice — not comparable to Tang static 18.9 dB without per-subject calibration.

## Full train (wide + SE-ResNet, 40 epochs)

```bash
openalterego train --data sessions/gaddy_cv --fs 1000 --preprocess-mode streaming \
  --emg-mode wide --arch se_resnet --epochs 40 --segment-ms 600
```

| Metric | Value |
|--------|-------|
| Best val accuracy | **25.0%** (2× chance on 8-way) |
| Final train accuracy | ~98% (overfit) |
| Checkpoint | `sessions/gaddy_cv/model.pt` |

## Interpretation

1. **Pipeline works end-to-end** on published human EMG (download → session → train → checkpoint).
2. **Wide bandpass + SE-ResNet** fits training data much better than standard on this corpus (consistent with Tang 2025 / literature direction).
3. **Closed-vocab accuracy is low** because:
   - Gaddy is **open-vocabulary silent speech** (book reading), not 6–10 command tokens.
   - We derived pseudo-labels from first words of sentences (days/months) — high intra-class variance.
   - Small val split (16 segments) + single-speaker session → unstable metrics.
4. **Next validation steps** (roadmap A):
   - Import **Gowda small-vocab** (67 words, 5000 Hz, 31 ch) for apples-to-apples closed-vocab test.
   - Collect **OpenAlterEgo hardware** sessions with command labels.
   - Use **WER/CTC** metrics for Gaddy open vocab (Phase E), not CNN token accuracy.

## Reproduce

```bash
cd software/python
openalterego dataset import-gaddy --raw ./datasets/gaddy_download/raw \
  --out ./sessions/gaddy_cv --max-samples 250 --top-labels 8 --min-samples-per-label 3 --skip-download
openalterego dataset ab-preprocess --session ./sessions/gaddy_cv --epochs 25 --json
openalterego train --data ./sessions/gaddy_cv --fs 1000 --preprocess-mode streaming --emg-mode wide --arch se_resnet
```
