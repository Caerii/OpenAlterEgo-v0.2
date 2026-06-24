# Gowda / emg2speech Documentation Index

Systematic documentation for **Gowda et al.** orofacial EMG silent-speech research, datasets, and OpenAlterEgo validation.

**Primary paper (2025):** [Non-invasive electromyographic speech neuroprosthesis: a geometric perspective](https://arxiv.org/html/2502.05762v2) — arXiv:2502.05762.

**Companion paper (2024):** [Geometry of orofacial neuromuscular signals](https://arxiv.org/abs/2411.02591) — SPD manifold foundations.

**Code & checkpoints:** [HarshavardhanaTG/emg2speech](https://github.com/HarshavardhanaTG/emg2speech)  
**Data:** [OSF YM5JD](https://doi.org/10.17605/OSF.IO/YM5JD) · [OSF box (bgh7t)](https://osf.io/bgh7t/files/box)

---

## Document map

| Section | Purpose |
|---------|---------|
| [papers/](papers/00-README.md) | Paper summaries, key claims, metrics |
| [datasets/](datasets/00-README.md) | OSF corpora, file formats, splits |
| [methods/](methods/00-README.md) | SPD features, preprocessing, models |
| [legal/](legal/00-README.md) | Ethics, licensing, IP, citations |
| [validation/](validation/00-README.md) | OpenAlterEgo benchmark results |
| [openalterego/](openalterego/00-README.md) | Import pipeline, gaps vs paper |

---

## Quick reference (2025 paper)

| Parameter | Large-vocab (§4) | Small-vocab (App. C.1) |
|-----------|------------------|------------------------|
| Channels | 31 @ 5 kHz | 31 @ 5 kHz |
| Task | EMG → phoneme sequence (CTC) | EMG → phoneme sequence (CTC) |
| Features | σ(τ) SPD matrices, 31×31 | σ(τ) SPD matrices, 31×31 |
| Window / hop | 50 ms context, **20 ms** step | 100 ms context, **50 ms** step |
| Train / val / test | 8000 / 1000 / 1970 **sentences** | 370 / 30 / 100 **sentences** |
| Reported metrics | PER **48.5%**, WER **73.5%** | PER **13%**, WER **14%** |
| Preprocessing | Ref subtract, 80–1000 Hz BP, per-ch z-norm | Same (App. B) |

---

## OpenAlterEgo status (June 2026)

| Item | Status |
|------|--------|
| OSF small-vocab import | ✅ `openalterego dataset import-gowda` |
| Corrected word boundaries + event alignment | ✅ June 2026 fix |
| Word-level CNN baseline (30 classes) | ✅ 17.8% val (trial split), 19.7% (Gowda split) |
| SPD + GRU + CTC (paper pipeline) | ✅ Phases 1–6 (~6.8% test WER) |
| Phoneme labels / PER-WER eval | ✅ |

See [validation/02-top30-corrected.md](validation/02-top30-corrected.md) and [openalterego/01-gap-analysis.md](openalterego/01-gap-analysis.md).

---

## Related repo docs

- Global references: [`../12-references.md`](../12-references.md) (§G1, G2) · [`../literature/README.md`](../literature/README.md)
- Roadmap: [`../14-systematic-roadmap.md`](../14-systematic-roadmap.md)
- User commands: [`../USER_GUIDE.md`](../USER_GUIDE.md)
