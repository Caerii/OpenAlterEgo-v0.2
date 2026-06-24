# Import Pipeline (OpenAlterEgo)

---

## CLI commands

```bash
cd software/python

# Small-vocab (auto-download OSF)
openalterego dataset import-gowda --download \
  --download-dir ./sessions/gowda_download \
  --out ./sessions/gowda_top30 \
  --top-labels 30 --min-samples-per-label 8

# Local npy files
openalterego dataset import-gowda \
  --data-npy ./sessions/gowda_download/dataSmallVocab.npy \
  --labels-npy ./sessions/gowda_download/labelsSmallVocab.npy \
  --out ./sessions/gowda_top30

# NATO subject folder
openalterego dataset import-gowda-nato \
  --subject-dir "./DATA/Subject 1" --out ./sessions/gowda_nato_s1
```

---

## Code map

| Module | Role |
|--------|------|
| `ml/datasets/gowda.py` | Download, trial cube import, word slicing, z-score |
| `ml/datasets/session.py` | `signals.npy` + `events.csv` + `meta.json` |
| `ml/data_split.py` | `gowda_sentence_train_val_indices()` for `--split-by gowda` |
| `ml/train.py` | Training with preprocess + segment cache, AMP, workers |
| `dsp/preprocess_cache.py` | Disk cache for streaming DSP |
| `ml/segment_cache.py` | Disk cache for `(N,C,T)` training windows |
| `ml/training_perf.py` | DataLoader defaults, optional `torch.compile` |

---

## Constants (`gowda.py`)

```python
GOWDA_TRIAL_SAMPLES = 45000
GOWDA_WORD_SLICE_SAMPLES = (10000, 10000, 10000, 15000)
GOWDA_SMALL_VOCAB_FS_HZ = 5000.0
```

---

## Session layout

```
sessions/gowda_top30/
  signals.npy          # (time, 31) float32, concatenated word segments
  events.csv           # start_sample, end_sample, label, trial_id, word_idx
  meta.json            # fs_hz, channels, source, notes
  preprocess_cache/    # optional streaming wide cache
  segments/            # optional (N,C,T) segment tensors per split/seed
  model_group_split.pt # best trial-split checkpoint
  model_gowda_split.pt # best gowda-split checkpoint
```

**Invariant:** `events[i].end - events[i].start` equals segment length; `events[i].start == sum(prev lengths)`.

---

## Train / eval

```bash
openalterego dataset cache-preprocess --session ./sessions/gowda_top30 \
  --emg-mode wide --preprocess-mode streaming --fs 5000

openalterego train --data ./sessions/gowda_top30 --fs 5000 \
  --preprocess-mode streaming --emg-mode wide --arch se_resnet \
  --epochs 30 --segment-ms 900 --split-by auto --batch-size 64

# AMP on by default on CUDA; --no-amp to disable
#   --rebuild-segment-cache  --num-workers 4  --compile  --mmap-signals

openalterego analyze channel-importance \
  --session ./sessions/gowda_top30 \
  --model ./sessions/gowda_top30/model_group_split.pt --top-k 16
```

---

## Tests

- `tests/test_model_datasets.py::TestGowdaImport` — alignment + slice lengths
- `tests/test_data_split.py::TestGowdaSentenceSplit` — 370/30 split
- `tests/test_segment_cache.py` — segment NPZ cache hit/invalidation
- `tests/test_training_perf.py` — DataLoader / compile helpers
