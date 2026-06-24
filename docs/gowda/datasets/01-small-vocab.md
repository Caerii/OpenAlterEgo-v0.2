# Dataset: Small-Vocabulary (Data<sub>small-vocab</sub>)

Paper: [arXiv:2502.05762v2](https://arxiv.org/html/2502.05762v2) Appendix C.1  
OSF files: `dataSmallVocab.npy`, `labelsSmallVocab.npy`

---

## Summary

| Field | Value |
|-------|-------|
| Unique words | **67** (weekdays, ordinals, months, years) |
| Sentences | **500** (paper); **499** in released `labelsSmallVocab[:499]` per emg2speech notebook |
| Participant | **1** (silent articulation) |
| Channels | **31** |
| Sample rate | **5000 Hz** |
| Trial length | **45 000** samples (9 s) per sentence |
| Word timing | **Timestamped** per word (paper); OSF release uses fixed slot layout |

---

## Sentence structure

Four words per sentence with on-screen timing:

| Slot | Duration | Example type |
|------|----------|----------------|
| 1 | 2 s | weekday |
| 2 | 2 s | month |
| 3 | 2 s | ordinal date |
| 4 | 3 s | year |

**Official emg2speech segmentation** (samples @ 5 kHz):

```
word0: [0 : 10000)
word1: [10000 : 20000)
word2: [20000 : 30000)
word3: [30000 : 45000)   # 15000 samples
```

OpenAlterEgo importer uses these boundaries (`GOWDA_WORD_SLICE_SAMPLES` in `gowda.py`).

---

## Array layout

```
dataSmallVocab.npy   shape (499, 31, 45000)   # (trials, channels, time)
labelsSmallVocab.npy shape (499, 4)            # string word per slot
```

---

## Paper train / val / test

| Split | Sentences | Word events (×4) |
|-------|-----------|------------------|
| Train | 370 | 1480 |
| Val | 30 | 120 |
| Test | 100 | 400 |

**Reported metrics (GRU + σ(τ) + CTC):** PER **13%**, WER **14%**.

---

## OpenAlterEgo sessions

| Session | Description |
|---------|-------------|
| `sessions/gowda_sv` | Early subset (200 trials, 12 labels) — **buggy import** era |
| `sessions/gowda_top30` | Full 499 trials, top-30 words, corrected import |

Import:

```bash
openalterego dataset import-gowda --download \
  --download-dir ./sessions/gowda_download \
  --out ./sessions/gowda_top30 \
  --top-labels 30 --min-samples-per-label 8
```

Validation: [../validation/02-top30-corrected.md](../validation/02-top30-corrected.md).

---

## Note on word-level CNN vs paper

The paper decodes **phoneme sequences** with CTC. OpenAlterEgo currently trains **word-level classification** (SE-ResNet CNN) — comparable only in spirit, not in PER/WER. See [../openalterego/01-gap-analysis.md](../openalterego/01-gap-analysis.md).
