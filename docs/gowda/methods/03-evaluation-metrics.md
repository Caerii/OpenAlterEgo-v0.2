# Evaluation Metrics & Splits

---

## Metrics (paper)

| Metric | Definition | Used on |
|--------|------------|---------|
| **PER** | Phoneme error rate — Levenshtein on phoneme sequences | Large + small vocab |
| **WER** | Word error rate — Levenshtein on words; HLG decoder (large) or 67-word lexicon match (small) | Large + small vocab |
| **CER** | Character error rate | NATO words only |

Chance levels (paper):

- PER ~97.5% (40 phonemes, uniform)
- CER ~96% (NATO alphabet)

---

## Splits

### Large-vocabulary (§4)

| Set | Sentences |
|-----|-----------|
| Train | 8000 |
| Val | 1000 |
| Test | 1970 |

### Small-vocabulary (App. C.1)

| Set | Sentences | Word events |
|-----|-----------|-------------|
| Train | 370 | 1480 |
| Val | 30 | 120 |
| Test | 100 | 400 |

### NATO words (App. C.2)

| Set | Articulations |
|-----|---------------|
| Train | 416 |
| Val | 104 |
| Test | 1968 |

---

## OpenAlterEgo metrics (different task)

| Metric | Definition |
|--------|------------|
| **Val accuracy** | Top-1 word classification on held-out **events** |
| Split `auto` | Stratified **trial_id** groups (no trial leakage) |
| Split `gowda` | First 370 / next 30 **trials** (paper sentence split) |

**Do not** report OpenAlterEgo accuracy as PER/WER without implementing phoneme CTC.

---

## CLI

```bash
# Paper-aligned sentence split
openalterego train --data ./sessions/gowda_top30 --split-by gowda ...
```

Implementation: `gowda_sentence_train_val_indices()` in `openalterego/ml/data_split.py`.
