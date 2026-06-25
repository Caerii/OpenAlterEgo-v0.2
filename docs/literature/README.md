# Literature & Papers — Master Index

**Last updated:** June 2026

This is the **entry point** for all silent-speech / EMG research references in OpenAlterEgo. Use it when you need a paper link, a **local PDF**, an offline text extract, or to see how literature maps to code.

---

## Where everything lives

| Location | Scope | When to use |
|----------|--------|-------------|
| **[`papers/`](papers/README.md)** | **Local PDF library** (open-access mirrors) + paywall stubs | Offline reading; air-gapped work |
| **[`papers/manifest.yaml`](papers/manifest.yaml)** | Download catalog (IDs match `12-references.md`) | Add/refresh local copies |
| **[`scripts/download_papers.py`](scripts/download_papers.py)** | Fetch open PDFs from manifest | `python docs/literature/scripts/download_papers.py` |
| **[`../12-references.md`](../12-references.md)** | Full curated bibliography | Default reading list; BibTeX; parameter matrix |
| **[`../../hardware/07-references.md`](../../hardware/07-references.md)** | Electrodes, AFE, wearables, SNR, BOM citations | Hardware design decisions |
| **[`../gowda/`](../gowda/00-README.md)** | Gowda / emg2speech papers, OSF data, SPD+CTC methods | Reproducing ~7% WER small-vocab benchmark |
| **[`archive/`](archive/)** | Legacy HTML→Markdown extracts (also copied beside key PDFs) | Full-text grep without PDF tooling |
| **`software/python/openalterego/sim/literature.py`** | Numeric sim presets — **not** paper PDFs | Synthetic EMG realism tuning |

### Local PDF layout

```
docs/literature/papers/
├── README.md              # auto-generated index + download status
├── manifest.yaml          # source of truth for downloads
├── download-status.json   # machine-readable status
├── A-surveys/             # A1, A2
├── B-alterego/            # AlterEgo lineage (+ stubs for paywalled ACM/PMLR)
├── D-wearables/           # Wang, Tang, SilentWear, Nature Sensors 2026
├── G-ml-decoding/         # Gowda geometry + neuroprosthesis PDFs
├── H-open-vocab/          # Gaddy, MONA LISA, IITK
├── K-foundational/        # EMG-UKA, INTERSPEECH classics
└── hardware/              # Datasheets + electrode reviews
```

**Coverage (typical run):** ~18 open-access PDFs on disk; ~12 publisher-paywalled entries keep a `.source.md` stub with the official DOI/link (ACM IUI, IEEE TIM, ScienceDirect, some MDPI bot-blocked mirrors).

**Also linked from:** [`../00-README-IMPLEMENTATION.md`](../00-README-IMPLEMENTATION.md) · [`../14-systematic-roadmap.md`](../14-systematic-roadmap.md) · [`../19-open-vocab-and-sim2real.md`](../19-open-vocab-and-sim2real.md)

---

## Featured papers (quick access)

High-impact references for **open vocabulary**, **LLM-era decoding**, **sensing hardware**, and our **primary benchmark**.

### MONA LISA — cross-modal silent speech + LLM reranking (2024)

| | |
|---|---|
| **Title** | A Cross-Modal Approach to Silent Speech with LLM-Enhanced Recognition |
| **Authors** | Tyler Benster, Guy Wilson, Reshef Elisha, Francis R. Willett, Shaul Druckmann |
| **Key result** | Gaddy open-vocab silent WER **28.8% → 12.2%**; vocal EMG **23.3% → 3.7%**; LLM scoring (LISA) |
| **Local PDF** | [`papers/H-open-vocab/benster-2024-mona-lisa-cross-modal-llm.pdf`](papers/H-open-vocab/benster-2024-mona-lisa-cross-modal-llm.pdf) |
| **arXiv** | [2403.05583](https://arxiv.org/abs/2403.05583) · [PDF](https://arxiv.org/pdf/2403.05583) |
| **OpenAlterEgo** | [`12-references.md` §H4](../12-references.md#h4-a-cross-modal-approach-to-silent-speech-with-llm-enhanced-recognition--mona-lisa-2024) · M4+ LM rerank in [`19-open-vocab-and-sim2real.md`](../19-open-vocab-and-sim2real.md) |

---

### Gowda & Miller — EMG neuroprosthesis, geometric / SPD + CTC (2025)

| | |
|---|---|
| **Title** | Non-invasive electromyographic speech neuroprosthesis: a geometric perspective |
| **Authors** | Harshavardhana T. Gowda, Lee M. Miller (UC Davis) |
| **Key result** | Direct silent EMG → phonemes via σ(τ) SPD + GRU + CTC; small-vocab **~14% WER** |
| **Local PDF** | [`papers/G-ml-decoding/gowda-2025-emg-neuroprosthesis-geometric.pdf`](papers/G-ml-decoding/gowda-2025-emg-neuroprosthesis-geometric.pdf) |
| **Text extract** | [`papers/G-ml-decoding/gowda-2025-emg-neuroprosthesis-geometric.extract.md`](papers/G-ml-decoding/gowda-2025-emg-neuroprosthesis-geometric.extract.md) |
| **arXiv** | [2502.05762](https://arxiv.org/abs/2502.05762) · [HTML](https://arxiv.org/html/2502.05762v2) · [PDF](https://arxiv.org/pdf/2502.05762) |
| **Code / data** | [emg2speech](https://github.com/HarshavardhanaTG/emg2speech) · [OSF YM5JD](https://doi.org/10.17605/OSF.IO/YM5JD) |
| **OpenAlterEgo** | [`../gowda/validation/`](../gowda/validation/00-README.md) (Phases 1–6) · `ml/spd/`, `ml/ctc/` |

---

### Xu et al. — SSI taxonomy in the LLM era (2026)

| | |
|---|---|
| **Title** | Silent Speech Interfaces in the Era of Large Language Models: A Comprehensive Taxonomy and Systematic Review |
| **Local PDF** | [`papers/A-surveys/xu-2026-ssi-llm-taxonomy-review.pdf`](papers/A-surveys/xu-2026-ssi-llm-taxonomy-review.pdf) |
| **arXiv** | [2603.11877](https://arxiv.org/abs/2603.11877) · [PDF](https://arxiv.org/pdf/2603.11877) |
| **OpenAlterEgo** | [`12-references.md` §A2](../12-references.md#a2-silent-speech-interfaces-in-the-llm-era-2026) |

---

### Tang et al. — Sensing technologies for SSIs (2026, *Nature Sensors*)

| | |
|---|---|
| **Title** | Sensing technologies for silent speech interfaces |
| **Local PDF** | [`papers/D-wearables/tang-2026-nature-sensors-ssi-survey.pdf`](papers/D-wearables/tang-2026-nature-sensors-ssi-survey.pdf) |
| **Text extract** | [`papers/D-wearables/tang-2026-nature-sensors-ssi-survey.extract.md`](papers/D-wearables/tang-2026-nature-sensors-ssi-survey.extract.md) |
| **Nature** | [s44460-025-00010-2](https://www.nature.com/articles/s44460-025-00010-2) · [PDF](https://www.nature.com/articles/s44460-025-00010-2.pdf) |
| **OpenAlterEgo** | [`12-references.md` §D5](../12-references.md#d5-sensing-technologies-for-silent-speech-interfaces-2026) · [`hardware/07-references.md`](../../hardware/07-references.md) §H-S2 |

---

## Browse by topic (→ full bibliography)

| Topic | Section in `12-references.md` | Local PDF folder |
|-------|-------------------------------|------------------|
| Surveys & reviews | [§A](../12-references.md#a-surveys--reviews) | `papers/A-surveys/` |
| AlterEgo lineage | [§B](../12-references.md#b-foundation-alterego-lineage) | `papers/B-alterego/` |
| Wearables & electrodes | [§D](../12-references.md#d-wearable-hardware--form-factors) + hardware bib | `papers/D-wearables/` |
| Signal processing | [§F](../12-references.md#f-signal-processing--quality) | `papers/F-signal-processing/` |
| ML & Gowda geometry | [§G](../12-references.md#g-machine-learning--decoding) | `papers/G-ml-decoding/` |
| Open vocab & LLM decode | [§H](../12-references.md#h-open-vocabulary--sequence-decoding) | `papers/H-open-vocab/` |
| Datasets & benchmarks | [§J](../12-references.md#j-datasets--benchmarks) | — (data on OSF/Zenodo) |
| Implementation vs papers | [Parameter matrix](../12-references.md#parameter-synthesis-openalterego-vs-literature) | — |

---

## Adding or refreshing a local copy

1. Add an entry to [`papers/manifest.yaml`](papers/manifest.yaml) (match bibliography ID in `12-references.md`).
2. Run `python docs/literature/scripts/download_papers.py`.
3. Add a **full entry** to [`../12-references.md`](../12-references.md) if not already present.
4. Optional: add `.extract.md` beside the PDF for grep-friendly full text.
5. If hardware-specific → also [`../../hardware/07-references.md`](../../hardware/07-references.md).

**Paywalled publishers** (ACM, IEEE, Elsevier): we store a `.source.md` stub with the official link — do not commit unauthorized PDFs. Add an open mirror URL to `manifest.yaml` only when legally available (author preprint, arXiv, institutional repository).
