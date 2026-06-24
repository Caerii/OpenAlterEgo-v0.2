# References: Silent Speech Interface Research

Curated bibliography for **OpenAlterEgo** — organized by topic, with explicit links to implementation decisions.

**Last updated:** June 2026

**Master index:** [`literature/README.md`](literature/README.md) — maps all bibliography locations, featured papers, and offline extracts.

---

## How to use this document

| Section | Use when… |
|---------|-----------|
| [Literature hub](literature/README.md) | Finding any paper quickly; offline archive paths |
| [Surveys](#a-surveys--reviews) | Onboarding; scoping modalities and clinical goals |
| [AlterEgo lineage](#b-foundation-alterego-lineage) | Validating our closed-vocabulary, wearable EMG approach |
| [Wearable & electrodes](#d-wearable-hardware--form-factors) | Hardware BOM, electrode count, dry vs wet |
| [Signal processing](#f-signal-processing--quality) | `dsp/filters.py`, `dsp/quality.py`, preprocessing modes |
| [ML & decoding](#g-machine-learning--decoding) | `ml/model.py`, architecture roadmap |
| [Open vocabulary](#h-open-vocabulary--sequence-decoding) | Future work beyond command tokens |
| [Datasets](#j-datasets--benchmarks) | External validation, sim realism targets |
| [Parameter matrix](#parameter-synthesis-openalterego-vs-literature) | Quick A/B of our defaults vs papers |

**Related docs:** `11-priority-changes.md`, `TODO.md`, `USER_GUIDE.md`, `13-data-collection-priorities`, [`hardware/README.md`](../hardware/README.md)

---

## A. Surveys & Reviews

### A1. Silent Speech Interfaces for Speech Restoration: A Review (2020)

**Authors:** José A. González-López, Alejandro Gómez-Alanis, Juan M. Martín Doñas, José Luis Pérez-Córdoba, Ángel M. Gómez  
**Venue:** IEEE Access, Vol. 8, pp. 177995–178021  
**DOI:** [10.1109/ACCESS.2020.3026579](https://doi.org/10.1109/ACCESS.2020.3026579) | [PDF](https://www.ugr.es/~joseangl/publication/gonzalez-lopez-silent-2020/gonzalez-lopez-silent-2020.pdf)

**Why important:** Best single entry point for SSI landscape — modalities (EMG, EEG, ultrasound, EMA), decode-to-text vs synthesize-speech, clinical vs HCI use cases, and the gap between lab demos and real-world deployment.

**Relevance:** Frames OpenAlterEgo as an **EMG command interface** within a broader SSI ecosystem; motivates motion robustness and clinical validation.

---

### A2. Silent Speech Interfaces in the Era of Large Language Models (2026)

**Authors:** Kele Xu, Yifan Wang, Ming Feng, Qisheng Xu, Wuyang Chen, Yutao Dou, Cheng Yang, Huaimin Wang  
**Venue:** arXiv preprint (systematic review, 20 pp.)  
**Links:** [arXiv:2603.11877](https://arxiv.org/abs/2603.11877) · [PDF](https://arxiv.org/pdf/2603.11877) · [Hub entry](literature/README.md#xu-et-al--ssi-taxonomy-in-the-llm-era-2026)

**Why important:** Up-to-date taxonomy across sensing (neural, EMG, ultrasound, RF), decode paradigms, and the shift from heuristic DSP to **latent semantic alignment** with LLMs as linguistic priors. Positions sub-15% WER as a usability threshold for open vocabulary.

**Relevance:** Validates our dual-mode roadmap (commands + open speech) and M3–M5 plan (personal LM, KenLM/LLM fusion) in [`19-open-vocab-and-sim2real.md`](19-open-vocab-and-sim2real.md).

---

## B. Foundation: AlterEgo Lineage

### B1. AlterEgo: A Personalized Wearable Silent Speech Interface (2018)

**Authors:** Arnav Kapur, Shreyas Kapur, Pattie Maes  
**Venue:** IUI 2018  
**Links:** [ACM](https://dl.acm.org/doi/10.1145/3172944.3172977) | [MIT Media Lab](https://www.media.mit.edu/projects/alterego/overview/)

**Key parameters:** 250 Hz; 1.3–50 Hz bandpass; 7 channels; 1D CNN; ~92% median word accuracy; BLE transport.

**Relevance:** Primary design ancestor. Validates **standard** preprocessing mode (`1–50 Hz`), per-user personalization, and closed-vocabulary wearable HCI.

---

### B2. Non-Invasive Silent Speech Recognition in Dysphonic Multiple Sclerosis (2020)

**Authors:** Arnav Kapur, Utkarsh Sarawgi, Eric Wadkins, Matthew Wu, Nora Hollenstein, Pattie Maes  
**Venue:** ML4H / PMLR  
**Link:** [PMLR](https://proceedings.mlr.press/v136/kapur20a.html)

**Key parameters:** 250 Hz; **0.5–8 Hz** clinical bandpass; 8 channels (4 face + 4 neck); earlobe reference; heartbeat wavelet removal.

**Relevance:** Validates **clinical** preprocessing mode; clinical population personalization.

---

### B3. Methods and Apparatus for Silent Speech Interface (Patent)

**Inventors:** Arnav Kapur, Shreyas Kapur, Pattie Maes  
**Patent:** US10878818B2  
**Link:** [Google Patents](https://patents.google.com/patent/US10878818B2)

**Relevance:** System architecture, safety/isolation, packet framing — guides `acquisition/packet.py` and hardware docs.

---

## C. Clinical & Assistive Applications

### C1. Electrode Setup for EMG-Based Silent Speech Interfaces: A Pilot Study (2025)

**Authors:** (MDPI Sensors) — Spanish laryngectomy database pilot  
**Venue:** Sensors 25(3):781  
**Link:** [MDPI](https://www.mdpi.com/1424-8220/25/3/781)

**Key finding:** Eight bipolar channels on digastric, DAO, risorius, levator labii, masseter, zygomaticus, depressor labii, stylohyoid — selected by phone-classification pilot.

**Relevance:** Electrode placement guide for future `docs/electrode-placement.md`; aligns with our 7–8 channel face/neck layout.

---

## D. Wearable Hardware & Form Factors

### D1. All-weather, natural silent speech recognition via tattoo-like electronics (2021)

**Authors:** Youhua Wang et al.  
**Venue:** npj Flexible Electronics 5:20  
**DOI:** [10.1038/s41528-021-00119-7](https://doi.org/10.1038/s41528-021-00119-7)

**Key parameters:** 4 channels (LAO, DAO, BUC, ABD); **500 Hz**; **20–500 Hz** bandpass; 2000 ms windows; LDA; **92.64%** on 110 words; 10 reps/word.

**Relevance:** Motivates **wide** bandpass; shows few channels + small data can work; long-wear robustness target.

---

### D2. Wireless Silent Speech Interface Using Multi-Channel Textile EMG in Headphones (2024/2025)

**Authors:** Chenyu Tang, Josée Mallah, Dominika Kazieczko, Wentian Yi, Tharun Reddy Kandukuri, Edoardo Occhipinti, Bhaskar Mishra, Sunita Mehta, Luigi G. Occhipinti  
**Venue:** IEEE Transactions on Instrumentation and Measurement  
**arXiv:** [2504.13921](https://arxiv.org/abs/2504.13921)

**Key parameters:** 4 textile channels; **1000 Hz**; **20–450 Hz** Butterworth; 3000 ms windows; **1D SE-ResNet**; **96%** on 10 commands; static SNR **18.9 dB** → motion **12.7 dB**.

**Relevance:** Primary citation for motion artifacts, wide bandpass, adaptive channel weighting (future SE blocks), and SNR baselines in `dsp/quality.py`.

---

### D3. SilentWear: Ultra-Low Power Wearable EMG Silent Speech (2025)

**Authors:** F. Meier, G. Spacone, S. Frey, L. Benini, A. Cossettini  
**Venue:** arXiv preprint / IEEE SENSORS 2025  
**arXiv:** [2603.02847](https://arxiv.org/abs/2603.02847)

**Key parameters:** Textile neck interface; **SpeechNet** (~15k params); 8 HMI commands; 4 subjects, multi-day; **84.8±4.6%** vocalized / **77.5±6.6%** silent (CV); inter-session **59–71%**.

**Relevance:** Edge deployment target (tiny CNN, on-device); multi-day drift / re-calibration requirements; validates compact models.

---

### D5. Sensing technologies for silent speech interfaces (2026)

**Authors:** Chenyu Tang, Liang Qi, Shuo Gao, Zibo Zhang, Wentian Yi, Muzi Xu, Edoardo Occhipinti, Yu Pan, Luigi G. Occhipinti  
**Venue:** *Nature Sensors* **1**, 16–26 (2026)  
**Links:** [Nature](https://www.nature.com/articles/s44460-025-00010-2) · [PDF](https://www.nature.com/articles/s44460-025-00010-2.pdf) · [Offline extract](literature/archive/tang-2026-nature-sensors-sensing-technologies.md) · [Hub entry](literature/README.md#tang-et-al--sensing-technologies-for-ssis-2026-nature-sensors)

**Key themes:** Off- / on- / in-body sensing continuum; flexible bioelectronics; multimodal fusion; edge AI; **on-body EMG** as accuracy vs deployability balance.

**Relevance:** Hardware north star for wearable form factors; complements D2 (Tang headphone EMG) and [`hardware/07-references.md`](../hardware/07-references.md) §H-S2.

---

## E. Electrode Placement & High-Density sEMG

### E1. Towards optimizing electrode configurations for SSR based on HD-sEMG (2021)

**Authors:** Ji et al.  
**Venue:** Journal of Neural Engineering  
**DOI:** [10.1088/1741-2552/abca14](https://doi.org/10.1088/1741-2552/abca14)

**Key finding:** Neck electrodes contribute more than face for digit recognition; **~10 optimal channels** can exceed dense arrays; English needs more channels than Chinese for same accuracy.

**Relevance:** Channel selection / redundancy analysis; supports 4–8 channel wearable designs.

---

### E2. Silent Speech Recognition Using Few Electrode Sites Guided by HD Arrays (2023)

**Authors:** Zhihang Deng, Xu Zhang, Xun Chen, Xiang Chen, Xi Chen, Erwei Yin  
**Venue:** IEEE Transactions on Instrumentation and Measurement, Vol. 72  
**DOI:** [10.1109/TIM.2023.3276540](https://doi.org/10.1109/TIM.2023.3276540)

**Key finding:** Transfer knowledge from 32×2 HD arrays to **8 sparse sites** (ZYG, RIS, DAO, SCM, ABD, PLT, …).

**Relevance:** Future channel-importance tooling; justifies our muscle-targeted montage.

---

### E3. HD-sEMG Silent Speech with Hybrid Networks & Anomaly Detection (2023)

**Authors:** Xu Zhang, Xun Chen, Xiang Chen, Le Wu, Yong Sun, Yuanfei Xia, Xi Chen  
**Venue:** IEEE Trans. Human-Machine Systems  
**DOI:** [10.1109/THMS.2022.3226197](https://doi.org/10.1109/THMS.2022.3226197)

**Key parameters:** 64 HD channels; 11 subjects; 33 Chinese words; CNN-LSTM + autoencoder; **82.3%** word accuracy, **90.6%** anomaly detection.

**Relevance:** Anomaly / non-speech rejection patterns for abstention logic in `runtime/streaming.py`.

---

## F. Signal Processing & Quality

### F1. Knowledge Distilled Ensemble Model for sEMG-based SSI (2023)

**Authors:** Wenqiang Lai, Qihan Yang, Ye Mao, Endong Sun, Jiangnan Ye  
**arXiv:** [2308.06533](https://arxiv.org/abs/2308.06533) | Code: [GitHub](https://github.com/laiwenq/AML_Lymsy)

**Key parameters:** 3 channels; **1000 Hz**; **20–400 Hz** (10th-order Butterworth); 1500 ms windows; ResNet1D ensemble + distillation; **85.9%** NATO alphabet.

**Relevance:** Validates ResNet1D family; distillation roadmap; wide bandpass consensus.

---

### F2. sEMG-based technology for silent voice recognition (2022)

**Authors:** (Computers in Biology and Medicine)  
**Link:** [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0010482522010447)

**Key parameters:** 10 Mandarin digits; STE-based VAD; SVM on spectral features; **92.3%** avg accuracy.

**Relevance:** Simple VAD / isolated-word baseline; STE as optional segmenter.

---

### F3. Design and implementation of a silent speech system based on sEMG (2024)

**Venue:** Biomedical Signal Processing and Control  
**Link:** [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1746809424001101)

**Relevance:** End-to-end deep learning pipeline survey point; ResNet/GRU comparisons in recent engineering literature.

---

## G. Machine Learning & Decoding

### G1. Geometry of orofacial neuromuscular signals (2024)

**Authors:** Harshavardhana T. Gowda, Zachary D. McNaughton, Lee M. Miller  
**Venue:** arXiv → Journal of Neural Engineering  
**arXiv:** [2411.02591](https://arxiv.org/abs/2411.02591) | Data: [OSF YM5JD](https://doi.org/10.17605/OSF.IO/YM5JD)

**Key parameters:** 16 subjects, **22 channels**, **5000 Hz**; SPD / Riemannian representations; small nets (≈10k–150k params); per-user eigenbasis shift.

**Relevance:** Theoretical basis for **per-user calibration**; largest open orofacial EMG corpus; future geometric priors.

---

### G2. Non-invasive EMG speech neuroprosthesis: a geometric perspective (2025)

**Authors:** Harshavardhana T. Gowda, Lee M. Miller  
**Links:** [arXiv:2502.05762](https://arxiv.org/abs/2502.05762) · [HTML](https://arxiv.org/html/2502.05762v2) · [PDF](https://arxiv.org/pdf/2502.05762) · [Code](https://github.com/HarshavardhanaTG/emg2speech) · [OSF YM5JD](https://doi.org/10.17605/OSF.IO/YM5JD)

**Local docs:** [`gowda/papers/01-neuroprosthesis-2025.md`](gowda/papers/01-neuroprosthesis-2025.md) · [Offline extract](literature/archive/gowda-2025-emg-neuroprosthesis-geometric.md) · [`gowda/00-README.md`](gowda/00-README.md)

**Key finding:** Direct **EMG→phoneme/text** via SPD matrices σ(τ) + GRU + **CTC** (no audio alignment); 31 electrodes @ 5 kHz; small-vocab WER **14%**, large-vocab PER **48.5%**.

**OpenAlterEgo validation:** Phases 1–6 in [`gowda/validation/`](gowda/validation/00-README.md) — **~6.8% test WER** with trial LM (June 2026).

**Relevance:** Primary open-vocabulary benchmark; `ml/spd/`, `ml/ctc/`, Gowda sim scenario (`sim/scenarios/gowda_small_vocab.py`).

---

## H. Open Vocabulary & Sequence Decoding

### H1. Digital Voicing of Silent Speech (2020)

**Authors:** David Gaddy, Dan Klein  
**Venue:** EMNLP 2020  
**Link:** [ACL Anthology](https://aclanthology.org/2020.emnlp-main.445/) | Code: [dgaddy/silent_speech](https://github.com/dgaddy/silent_speech)

**Key parameters:** ~20 h facial EMG + audio; 8 channels @ 1000 Hz; silent→audio synthesis; CTC recognition ~28% WER.

**Relevance:** Gaddy benchmark; cross-modal (vocalized+silent) training paradigm.

---

### H2. An Improved Model for Voicing Silent Speech (2021)

**Authors:** David Gaddy, Dan Klein  
**Venue:** ACL-IJCNLP 2021 (Short)  
**Link:** [ACL Anthology](https://aclanthology.org/2021.acl-short.23/)

**Relevance:** Improved synthesis; same dataset ecosystem.

---

### H3. Voicing Silent Speech (Dissertation, 2022)

**Author:** David Gaddy  
**Link:** [Berkeley Tech Report EECS-2022-68](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2022/EECS-2022-68.pdf)

**Relevance:** Comprehensive treatment of EMG speech recognition + synthesis; WER baselines.

---

### H4. A Cross-Modal Approach to Silent Speech with LLM-Enhanced Recognition — MONA LISA (2024)

**Authors:** Tyler Benster, Guy Wilson, Reshef Elisha, Francis R. Willett, Shaul Druckmann  
**Links:** [Stanford HAI](https://hai.stanford.edu/research/a-cross-modal-approach-to-silent-speech-with-llm-enhanced-recognition) · [arXiv:2403.05583](https://arxiv.org/abs/2403.05583) · [PDF](https://arxiv.org/pdf/2403.05583) · [Hub entry](literature/README.md#mona-lisa--cross-modal-silent-speech--llm-reranking-2024)

**Key finding:** MONA (cross-contrast + supTcon) aligns silent EMG with audio pretraining; **LISA** LLM scoring; Gaddy open-vocab silent WER **28.8% → 12.2%**, vocal EMG **23.3% → 3.7%**; first sub-15% open-vocab non-invasive silent speech at publication.

**Relevance:** Cross-modal sim2real pretraining; LLM/LM post-decode (M4 in [`19-open-vocab-and-sim2real.md`](19-open-vocab-and-sim2real.md)); complements Gaddy H1–H3.

---

### H5. Silent Speech Recognition using EMG — Seq2Seq sentence-level (IITK)

**Authors:** Amitangshu et al.  
**Link:** [IITK PDF](https://www.cse.iitk.ac.in/users/amitangshu/Silent_Speech_Recognition_using_Electromyography_Signals.pdf)

**Key finding:** Attention Seq2Seq on OpenBCI Cyton; **9.3% WER** on continuous sEMG sentences vs CNN+BiLSTM+CTC baseline.

**Relevance:** Sentence-level decoding beyond isolated tokens; OpenBCI compatibility reference.

---

## I. Multimodal & Related (non-EMG primary)

| Paper | Modality | Note |
|-------|----------|------|
| Sentence-Level SSR with wearable EMG+EEG + LM (2024) | EMG+EEG fusion | [MDPI Sensors 25(19):6168](https://www.mdpi.com/1424-8220/25/19/6168) — 95% sentence accuracy, 10-word military commands |
| Kimura et al. (2019) | Ultrasound | Silent speech via articulatory imaging |
| Willett et al. (2023) | Intracortical BCI | High-bandwidth speech BCI benchmark (invasive) |

**Relevance:** IMU/EEG fusion and multimodal context are post-MVP (`TODO.md`).

---

## J. Datasets & Benchmarks

| Dataset | Size | Channels | fs | Modes | Link |
|---------|------|----------|-----|-------|------|
| **Gowda OSF (2024)** | 16 subjects | 22 | 5000 Hz | silent + audible, phonemes/words/NATO | [osf.io/ym5jd](https://osf.io/ym5jd/) |
| **Gaddy EMG (2020)** | ~20 h, 1 speaker | 8 | 1000 Hz | silent + vocal + audio | [silent_speech repo](https://github.com/dgaddy/silent_speech) |
| **EMG-UKA (2014)** | 7.5 h, 8 speakers | 6 | 600 Hz | audible, whisper, silent | [INTERSPEECH](https://www.isca-archive.org/interspeech_2014/wand14b_interspeech.html) |
| **SilentWear (2025)** | 4 subjects, multi-day | textile neck | — | 8 commands | [arXiv:2603.02847](https://arxiv.org/abs/2603.02847) |

**OpenAlterEgo sim targets:** `sim/literature.py` paradigms (`alterego_envelope`, `semg_literature_clamped`, `semg_literature_full`).

---

## K. Foundational Historical Work

### K1. Towards continuous speech recognition using surface EMG (2006)

**Authors:** Szu-Chen Jou, Tanja Schultz, Matthias Walliczek, Florian Kraft, Alex Waibel  
**Venue:** INTERSPEECH 2006

**Relevance:** Early continuous EMG ASR; coarticulation challenges.

---

### K2. Modeling coarticulation in EMG-based continuous speech recognition (2010)

**Authors:** Tanja Schultz, Michael Wand  
**Venue:** Speech Communication 52(4)

**Relevance:** Coarticulation modeling — informs window length and phonology sim (`sim/phonology/`).

---

### K3. The EMG-UKA Corpus (2014)

**Authors:** Michael Wand, Matthias Janke, Tanja Schultz  
**Venue:** INTERSPEECH 2014

**Relevance:** First major public EMG speech corpus; HMM baselines.

---

### K4. Synthesizing speech from electromyography (2009)

**Authors:** Arthur R. Toth, Michael Wand, Tanja Schultz  
**Venue:** INTERSPEECH 2009

**Relevance:** EMG→acoustic synthesis lineage (alternative to text tokens).

---

## L. Software, Hardware & Standards

| Resource | Link | Use |
|----------|------|-----|
| **Hardware design (systematic)** | [`hardware/README.md`](../hardware/README.md) | AFE, electrodes, BLE, BOM, safety, mechanical |
| **Hardware bibliography** | [`hardware/07-references.md`](../hardware/07-references.md) | Electrode/AFE/wearable-specific papers |
| **Literature hub & offline archive** | [`literature/README.md`](literature/README.md) | Master index + searchable extracts |
| LibEMG | [github.com/LibEMG/libemg](https://github.com/LibEMG/libemg) | EMG processing utilities |
| dgaddy/silent_speech | [GitHub](https://github.com/dgaddy/silent_speech) | Gaddy data + baselines |
| OpenBCI | [openbci.com](https://openbci.com/) | V0 benchtop acquisition |
| ADS1299 | TI datasheet | AFE reference (`hardware/BOM.md`) |
| nRF52 BLE | Nordic | Wearable MCU target |
| IEC 60601 | — | Medical electrical safety (future clinical) |

---

## Parameter Synthesis: OpenAlterEgo vs Literature

| Parameter | OpenAlterEgo default | AlterEgo (2018) | Recent EMG SSI (2021–25) | Status |
|-----------|-------------------|-----------------|--------------------------|--------|
| Sampling rate | **250 Hz** | 250 Hz | 500–1000 Hz (up to 5 kHz research) | ✅ Intentional; configurable in sim |
| Bandpass | **standard** 1–50 Hz; **wide** 20–450 Hz; **clinical** 0.5–8 Hz | 1.3–50 Hz | 20–400/500 Hz | ✅ All three modes implemented |
| Channels | 7–8 | 7–8 | 3–4 often sufficient | ✅ Redundant by design |
| Window | **600 ms** | — | 1500–3000 ms | ⚠️ Shorter than literature; configurable |
| Stride | **120 ms** | — | 100–400 ms sliding | ✅ OK for commands |
| Model | **OpenAlterEgoCNN** (1D CNN) | 1D CNN | ResNet1D, SE-ResNet, SpeechNet | ✅ Baseline OK; SE-ResNet future |
| Personalization | **Per-user calibrate/train/serve** | Per-user | Per-subject essential | ✅ Core path done |
| Motion / SNR | **`dsp/quality`**, online monitor | Limited | 33% SNR drop under motion | ✅ Detected; preprocessing gating optional |
| Vocabulary | **Closed commands** (~6 tokens) | ~100 words | Open vocab WER benchmarks | 🔮 Future (phoneme/seq2seq) |
| Transport | **BLE + WebSocket JSON** | BLE | WiFi/BLE mixed | ✅ Protocol in `api/protocol.py` |

---

## OpenAlterEgo Alignment Matrix

| Literature insight | Code location | Next action |
|--------------------|---------------|-------------|
| Wide bandpass 20–450 Hz | `dsp/filters.py`, `users/profile.py` | A/B eval on real data |
| Motion SNR degradation | `dsp/quality.py`, `users/calibration.py` | Optional preprocess gating |
| Per-user "change of basis" | `users/*`, `ml/train.py` | Re-calibration triggers |
| SE channel weighting | — | Add SE blocks to `ml/model.py` |
| Longer windows (1.5–3 s) | `runtime/streaming.py`, `UserProfile.window_ms` | Benchmark accuracy vs latency |
| HD electrode selection | — | Channel importance viz |
| Open vocab / phonemes | — | Phonology sim → seq2seq decoder |
| Cross-modal / LLM | — | Research track (MONA LISA) |
| Edge tiny models | `ml/model.py` | SpeechNet-scale distillation |

---

## BibTeX (core citations)

```bibtex
@inproceedings{kapur2018alterego,
  title={AlterEgo: A Personalized Wearable Silent Speech Interface},
  author={Kapur, Arnav and Kapur, Shreyas and Maes, Pattie},
  booktitle={IUI},
  year={2018}
}

@article{gonzalez2020ssi,
  title={Silent Speech Interfaces for Speech Restoration: A Review},
  author={Gonz{\'a}lez-L{\'o}pez, Jos{\'e} A. and others},
  journal={IEEE Access},
  volume={8},
  pages={177995--178021},
  year={2020}
}

@article{wang2021tattoo,
  title={All-weather, natural silent speech recognition via machine-learning-assisted tattoo-like electronics},
  author={Wang, Youhua and others},
  journal={npj Flexible Electronics},
  volume={5},
  number={20},
  year={2021}
}

@article{tang2025headphone,
  title={Wireless Silent Speech Interface Using Multi-Channel Textile EMG Sensors Integrated into Headphones},
  author={Tang, Chenyu and others},
  journal={IEEE Transactions on Instrumentation and Measurement},
  year={2025},
  note={arXiv:2504.13921}
}

@article{lai2023kd,
  title={Knowledge Distilled Ensemble Model for sEMG-based Silent Speech Interface},
  author={Lai, Wenqiang and others},
  journal={arXiv:2308.06533},
  year={2023}
}

@article{gowda2024geometry,
  title={Geometry of orofacial neuromuscular signals},
  author={Gowda, Harshavardhana T. and McNaughton, Zachary D. and Miller, Lee M.},
  journal={arXiv:2411.02591},
  year={2024}
}

@article{gowda2025neuroprosthesis,
  title={Non-invasive electromyographic speech neuroprosthesis: a geometric perspective},
  author={Gowda, Harshavardhana T. and Miller, Lee M.},
  journal={arXiv:2502.05762},
  year={2025}
}

@inproceedings{gaddy2020voicing,
  title={Digital Voicing of Silent Speech},
  author={Gaddy, David and Klein, Dan},
  booktitle={EMNLP},
  year={2020}
}

@article{benster2024mona,
  title={A Cross-Modal Approach to Silent Speech with LLM-Enhanced Recognition},
  author={Benster, Tyler and Wilson, Guy and Elisha, Reshef and Willett, Francis R. and Druckmann, Shaul},
  journal={arXiv:2403.05583},
  year={2024}
}

@article{xu2026ssi_llm_review,
  title={Silent Speech Interfaces in the Era of Large Language Models: A Comprehensive Taxonomy and Systematic Review},
  author={Xu, Kele and Wang, Yifan and Feng, Ming and Xu, Qisheng and Chen, Wuyang and Dou, Yutao and Yang, Cheng and Wang, Huaimin},
  journal={arXiv:2603.11877},
  year={2026}
}

@article{tang2026nature_sensors,
  title={Sensing technologies for silent speech interfaces},
  author={Tang, Chenyu and Qi, Liang and Gao, Shuo and others},
  journal={Nature Sensors},
  volume={1},
  pages={16--26},
  year={2026},
  doi={10.1038/s44460-025-00010-2}
}

@article{deng2023few,
  title={Silent Speech Recognition Based on sEMG Using Few Electrode Sites Under Guidance From HD Arrays},
  author={Deng, Zhihang and others},
  journal={IEEE Trans. Instrumentation and Measurement},
  volume={72},
  year={2023}
}

@article{meier2025silentwear,
  title={SilentWear: an Ultra-Low Power Wearable System for EMG-based Silent Speech},
  author={Meier, F. and others},
  journal={arXiv:2603.02847},
  year={2025}
}

@inproceedings{wand2014emguka,
  title={The EMG-UKA Corpus for Electromyographic Speech Processing},
  author={Wand, Michael and Janke, Matthias and Schultz, Tanja},
  booktitle={INTERSPEECH},
  year={2014}
}
```

---

## Summary: Papers that most directly drive our stack

1. **Kapur 2018 / 2020** — AlterEgo foundation; standard + clinical modes ✅  
2. **Wang 2021, Tang 2025, Lai 2023** — Wide bandpass, motion SNR, ResNet family ✅  
3. **Gowda 2024 / 2025** — Personalization theory + open dataset ✅  
4. **Deng 2023, Ji 2021** — Electrode count and placement 🔮  
5. **Meier 2025 (SilentWear)** — Edge tiny models, multi-day drift 🔮  
6. **Gaddy 2020 + MONA 2024** — Open vocabulary roadmap 🔮  
7. **González-López 2020** — Field-wide context and clinical framing 📚

**Key takeaway:** The **vertical slice** (sim → calibrate → train → serve) is implemented. The literature now pushes us toward **real-data evaluation**, **longer windows / SE architectures**, and eventually **phoneme- or sequence-level decoding**.
