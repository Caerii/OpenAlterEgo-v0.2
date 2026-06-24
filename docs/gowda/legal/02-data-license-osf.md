# Data License & OSF Access

---

## Primary repositories

| Resource | DOI / URL | Contents |
|----------|-----------|----------|
| **OSF project** | [10.17605/OSF.IO/YM5JD](https://doi.org/10.17605/OSF.IO/YM5JD) | Master project page |
| **File box (emg2speech)** | [osf.io/bgh7t/files/box](https://osf.io/bgh7t/files/box) | `dataSmallVocab.npy`, `labelsSmallVocab.npy`, large-vocab pickles, NATO subject folders |
| **GitHub** | [HarshavardhanaTG/emg2speech](https://github.com/HarshavardhanaTG/emg2speech) | Notebooks, checkpoints (DATA not in repo) |

---

## Files used by OpenAlterEgo

| File | Approx. size | Layout | Import command |
|------|--------------|--------|----------------|
| `dataSmallVocab.npy` | ~5.5 GB | `(499, 31, 45000)` trials, float | `openalterego dataset import-gowda --download` |
| `labelsSmallVocab.npy` | small | `(trials, 4)` word grid | paired with above |

**Direct OSF download URLs** (used by `gowda.py`):

- Data: `https://osf.io/download/cj9kb/`
- Labels: `https://osf.io/download/htpcg/`

---

## License practice

1. **Check OSF project “License” field** at download time — terms may update independently of this doc.
2. **Do not commit** raw `.npy` files to git (size + license). Store under `sessions/gowda_download/` (gitignored).
3. **Derived artifacts** (session folders, checkpoints, preprocess cache) are project outputs — document provenance in validation notes.
4. **Redistribution** of raw OSF files must comply with OSF/license terms; prefer linking to OSF DOI.

---

## Dataset subsets (paper naming)

| Name | Paper section | OpenAlterEgo session examples |
|------|---------------|-------------------------------|
| Data<sub>small-vocab</sub> | App. C.1 | `gowda_sv`, `gowda_top30` |
| Data<sub>nato-words</sub> | App. C.2 | `import-gowda-nato` (not fully benchmarked) |
| Large-vocab corpus | §4 | Not yet imported |

Details: [../datasets/00-README.md](../datasets/00-README.md).
