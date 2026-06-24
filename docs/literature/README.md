# Literature & Papers — Master Index

**Last updated:** June 2026

This is the **entry point** for all silent-speech / EMG research references in OpenAlterEgo. Use it when you need a paper link, an offline excerpt, or to see how literature maps to code.

---

## Where everything lives

| Location | Scope | When to use |
|----------|--------|-------------|
| **[`../12-references.md`](../12-references.md)** | Full curated bibliography (surveys → ML → open vocab → datasets) | Default reading list; BibTeX; parameter matrix |
| **[`../../hardware/07-references.md`](../../hardware/07-references.md)** | Electrodes, AFE, wearables, SNR, BOM citations | Hardware design decisions |
| **[`../gowda/`](../gowda/00-README.md)** | Gowda / emg2speech papers, OSF data, SPD+CTC methods, validation phases | Reproducing ~7% WER small-vocab benchmark |
| **[`archive/`](archive/)** | Offline HTML/Markdown extracts (searchable in-repo) | Air-gapped or quick full-text grep |
| **`software/python/openalterego/sim/literature.py`** | Numeric sim presets (bandpass, SNR targets) — **not** paper PDFs | Synthetic EMG realism tuning |

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
| **Stanford HAI** | [Project page](https://hai.stanford.edu/research/a-cross-modal-approach-to-silent-speech-with-llm-enhanced-recognition) |
| **arXiv** | [2403.05583](https://arxiv.org/abs/2403.05583) · [PDF](https://arxiv.org/pdf/2403.05583) |
| **OpenAlterEgo** | [`12-references.md` §H4](../12-references.md#h4-a-cross-modal-approach-to-silent-speech-with-llm-enhanced-recognition--mona-lisa-2024) · M4+ LM rerank in [`19-open-vocab-and-sim2real.md`](../19-open-vocab-and-sim2real.md) |

---

### Gowda & Miller — EMG neuroprosthesis, geometric / SPD + CTC (2025)

| | |
|---|---|
| **Title** | Non-invasive electromyographic speech neuroprosthesis: a geometric perspective |
| **Authors** | Harshavardhana T. Gowda, Lee M. Miller (UC Davis) |
| **Key result** | Direct silent EMG → phonemes via σ(τ) SPD + GRU + CTC; small-vocab **~14% WER** |
| **arXiv** | [2502.05762](https://arxiv.org/abs/2502.05762) · [HTML](https://arxiv.org/html/2502.05762v2) · [PDF](https://arxiv.org/pdf/2502.05762) |
| **Code / data** | [emg2speech](https://github.com/HarshavardhanaTG/emg2speech) · [OSF YM5JD](https://doi.org/10.17605/OSF.IO/YM5JD) |
| **Local summary** | [`../gowda/papers/01-neuroprosthesis-2025.md`](../gowda/papers/01-neuroprosthesis-2025.md) |
| **Offline extract** | [`archive/gowda-2025-emg-neuroprosthesis-geometric.md`](archive/gowda-2025-emg-neuroprosthesis-geometric.md) |
| **OpenAlterEgo** | [`../gowda/validation/`](../gowda/validation/00-README.md) (Phases 1–6) · `ml/spd/`, `ml/ctc/` |

---

### Xu et al. — SSI taxonomy in the LLM era (2026)

| | |
|---|---|
| **Title** | Silent Speech Interfaces in the Era of Large Language Models: A Comprehensive Taxonomy and Systematic Review |
| **Authors** | Kele Xu, Yifan Wang, Ming Feng, Qisheng Xu, Wuyang Chen, Yutao Dou, Cheng Yang, Huaimin Wang |
| **Key themes** | Latent semantic alignment; LLMs as linguistic priors; sensing modalities (neural → EMG → articulatory → RF); wearable “invisible interfaces”; neuro-security |
| **arXiv** | [2603.11877](https://arxiv.org/abs/2603.11877) · [PDF](https://arxiv.org/pdf/2603.11877) |
| **OpenAlterEgo** | [`12-references.md` §A2](../12-references.md#a2-silent-speech-interfaces-in-the-llm-era-2026) |

---

### Tang et al. — Sensing technologies for SSIs (2026, *Nature Sensors*)

| | |
|---|---|
| **Title** | Sensing technologies for silent speech interfaces |
| **Authors** | Chenyu Tang, Liang Qi, Shuo Gao, Zibo Zhang, Wentian Yi, Muzi Xu, Edoardo Occhipinti, Yu Pan, Luigi G. Occhipinti |
| **Venue** | *Nature Sensors* **1**, 16–26 (2026) |
| **Key themes** | Off- / on- / in-body sensing trade-offs; flexible bioelectronics; multimodal fusion; edge AI; on-body EMG as deployability sweet spot |
| **Nature** | [s44460-025-00010-2](https://www.nature.com/articles/s44460-025-00010-2) · [PDF](https://www.nature.com/articles/s44460-025-00010-2.pdf) |
| **Offline extract** | [`archive/tang-2026-nature-sensors-sensing-technologies.md`](archive/tang-2026-nature-sensors-sensing-technologies.md) |
| **OpenAlterEgo** | [`12-references.md` §D5](../12-references.md#d5-sensing-technologies-for-silent-speech-interfaces-2026) · [`hardware/07-references.md`](../../hardware/07-references.md) §H-S2 |

---

## Browse by topic (→ full bibliography)

| Topic | Section in `12-references.md` |
|-------|-------------------------------|
| Surveys & reviews | [§A](../12-references.md#a-surveys--reviews) |
| AlterEgo lineage | [§B](../12-references.md#b-foundation-alterego-lineage) |
| Wearables & electrodes | [§D](../12-references.md#d-wearable-hardware--form-factors) + hardware bib |
| Signal processing | [§F](../12-references.md#f-signal-processing--quality) |
| ML & Gowda geometry | [§G](../12-references.md#g-machine-learning--decoding) |
| Open vocab & LLM decode | [§H](../12-references.md#h-open-vocabulary--sequence-decoding) |
| Datasets & benchmarks | [§J](../12-references.md#j-datasets--benchmarks) |
| Implementation vs papers | [Parameter matrix](../12-references.md#parameter-synthesis-openalterego-vs-literature) |

---

## Adding a new paper

1. Add a **full entry** to [`../12-references.md`](../12-references.md) (correct section, DOI/arXiv/PDF links).
2. If hardware-specific → also [`../../hardware/07-references.md`](../../hardware/07-references.md).
3. If Gowda/emg2speech ecosystem → [`../gowda/papers/`](../gowda/papers/00-README.md).
4. Optional: drop HTML/Markdown extract in [`archive/`](archive/) and link from this file.
5. Cross-link from roadmap / gap analysis if it changes implementation priorities.
