# Dataset: NATO Words (Data<sub>nato-words</sub>)

Paper: [arXiv:2502.05762v2](https://arxiv.org/html/2502.05762v2) Appendix C.2

---

## Summary

| Field | Value |
|-------|-------|
| Subjects | **4** |
| Channels | **22** @ 5 kHz (no right-neck array) |
| Content | Rainbow + grandfather passages in **NATO spelled** form |
| Articulations | 1968 codeword tokens + 520 isolated training tokens |
| Split | 416 train / 104 val / **1968 test** articulations |
| Metric | **CER** (character error rate); chance ~96% |

**Reported CER:** 55–70% per subject (single-layer GRU, 100 epochs).

---

## Example

Word “cat” articulated as NATO codewords → phonemic sequence for “Charlie Alpha Tango”.

---

## Temporal features (paper)

| Parameter | Value |
|-----------|-------|
| Window context | 150 ms |
| Step τ | **30 ms** |
| σ(τ) size | **22×22** |

---

## OpenAlterEgo

```bash
openalterego dataset import-gowda-nato --subject-dir "./DATA/Subject 1" --out ./sessions/gowda_nato_s1
```

Not yet benchmarked to paper CER. Import adapter: `import_gowda_nato_subject()` in `gowda.py`.

---

## Use case

Minimal-data spelling interface — train ~10 min, test ~50 min per paper design.
