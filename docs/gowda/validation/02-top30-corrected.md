# Validation: Top-30 Corrected Import (`gowda_top30`)

June 2026 — full 499-trial import with fixed alignment, official word slices, per-trial z-score.

---

## Import fixes

| Issue | Before | After |
|-------|--------|-------|
| Event vs signal indices | Phantom 500-sample gaps | Contiguous `start_sample` |
| Word boundaries | Equal 11 250-sample quarters | **10k / 10k / 10k / 15k** |
| Normalization | None at import | Per-trial z-score per channel |

Code: `openalterego/ml/datasets/gowda.py`

---

## Session

| Field | Value |
|-------|-------|
| Events | 1 223 (top-30 labels, ≥8 samples/label) |
| Channels | 31 @ 5000 Hz |
| Duration | ~2 446 s |

---

## Training (SE-ResNet, wide, streaming, 900 ms, 30 epochs, GPU+AMP)

```bash
openalterego train --data ./sessions/gowda_top30 --fs 5000 \
  --preprocess-mode streaming --emg-mode wide --arch se_resnet \
  --epochs 30 --segment-ms 900 --batch-size 64 --amp
```

| Split | Train / val | Best val acc | Checkpoint |
|-------|-------------|--------------|------------|
| Trial (`--split-by auto`) | 976 / 247 | **17.8%** | `model_group_split.pt` |
| Gowda (`--split-by gowda`) | 908 / 76 | **19.7%** | `model_gowda_split.pt` |
| **Phase 1** (`gowda` + per-event + 2000 ms) | 908 / 76 | **36.8%** | `ablations/per_event_gowda_2000.pt` |

See [`03-phase1-ablation.md`](03-phase1-ablation.md) for the full matrix.  
Phase 2 (weekday ceiling, bootstrap CI, CTC): [`04-phase2-ceiling-ctc.md`](04-phase2-ceiling-ctc.md).

Chance (30-way): **3.3%**. Pre-fix buggy import: **7.2%**.

---

## vs paper (App. C.1)

| | Paper (GRU+σ+CTC) | OpenAlterEgo (CNN words) |
|--|-------------------|--------------------------|
| Metric | WER **14%** | Val acc **~18–20%** |
| Labels | Phoneme sequences | 30 word classes |
| Comparable? | **No** — different task/metrics |

---

## Channel importance

Top-16 (gradient saliency): `[3, 13, 2, 24, 0, 29, 30, 21, 20, 4, 16, 11, 12, 1, 27, 5]`

---

## Logs

- `sessions/gowda_top30/train_corrected_group.log`
- `sessions/gowda_top30/train_corrected_gowda.log`
