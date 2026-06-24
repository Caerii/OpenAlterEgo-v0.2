# Paper Summary: Non-invasive EMG Speech Neuroprosthesis (2025)

**Authors:** Harshavardhana T. Gowda, Lee M. Miller (UC Davis)  
**Link:** [arXiv:2502.05762v2](https://arxiv.org/html/2502.05762v2)  
**Code / data:** [emg2speech](https://github.com/HarshavardhanaTG/emg2speech) · [OSF YM5JD](https://doi.org/10.17605/OSF.IO/YM5JD)

---

## One-sentence claim

Silent orofacial EMG can be decoded **directly to phoneme sequences** using **SPD covariance features** and **CTC**, without time-aligned audio — achieving ~49% phoneme error rate on a large open-vocabulary English corpus.

---

## Problem addressed

| Prior approach (e.g. Gaddy 2020) | Limitation |
|----------------------------------|------------|
| EMG + audible speech + audio targets | Needs vocalized speech; 2× data collection |
| Time-aligned EMG–audio features | Expensive alignment; hard for real-time |
| Spectrogram + RNN | Poor phonemic structure (WER → 100% in their baseline) |

**This paper:** Train on **silent EMG only** (E<sub>S</sub>) with phoneme labels (L) via **CTC** — alignment-free sequence learning.

---

## Core method

### 1. Graph / SPD representation

For each time window τ:

1. Stack multichannel EMG → edge matrix ℰ(τ) with entries e<sub>ij</sub> = f<sub>i</sub><sup>T</sup>f<sub>j</sub> (channel covariance).
2. Regularize to SPD: ℰ ← (1−η)ℰ + η·trace(ℰ)·I, **η = 0.1**.
3. Compute **Fréchet mean** ℱ over training windows; fixed eigenbasis **Q** from eig(ℱ).
4. Approximate diagonalization: **σ(τ) = Q<sup>T</sup> ℰ(τ) Q** → sparse 31×31 input to GRU.

### 2. Sequence model

- **GRU** (up to 3 layers on large-vocab; 1 layer on small-vocab appendix).
- **CTC loss** → per-frame phoneme probabilities (40 phoneme labels).
- **Beam search** (width 50) for decoding.
- **HLG graph** (CTC ⊗ lexicon ⊗ 4-gram LM) for word-level WER on large corpus.

### 3. Temporal resolution

| Corpus | Context window | Step (τ) | σ(τ) size |
|--------|----------------|----------|-----------|
| Large-vocab (§4) | 50 ms overlap | **20 ms** | 31×31 |
| Small-vocab (App. C.1) | 100 ms overlap | **50 ms** | 31×31 |
| NATO words (App. C.2) | 150 ms overlap | **30 ms** | 22×22 |

---

## Acquisition (Appendix B)

| Parameter | Value |
|-----------|-------|
| Electrodes | **31** monopolar sites (neck, chin, jaw, cheek, lips) |
| Reference | Electrode **32** on right earlobe — subtracted from all channels |
| Ground | Left earlobe |
| Amplifier | Brain Vision actiCHamp Plus |
| Sample rate | **5000 Hz** |
| Gel | SuperVisc (Easycap) |
| Sync | LSL |
| Sentence markers | Subject **mouse click** at start and end of each sentence |

### Preprocessing (minimal)

1. Reference channel subtraction  
2. **3rd-order Butterworth bandpass 80–1000 Hz**  
3. Segment by sentence timestamps  
4. **Per-channel z-normalization along time** within segment  
5. Build ℰ(τ) and σ(τ)

---

## Main results (Table 1 — large-vocab)

| Model | PER ↓ | WER ↓ |
|-------|-------|-------|
| Spectrogram baseline | 89.25% | 100% |
| **σ(τ) (ours)** | **48.47%** | **73.53%** |

- Train / val / test: **8000 / 1000 / 1970 sentences** (~6500 unique words).
- Chance PER ≈ 97.5% (uniform over 40 phonemes).
- ~160 words/min speaking rate; Willett et al. sentence corpus adapted.

---

## Small-vocabulary appendix (C.1) — **OpenAlterEgo primary benchmark**

| Item | Detail |
|------|--------|
| Vocabulary | **67 words** (weekdays, ordinals, months, years) |
| Sentences | **500** silently articulated by **one** participant |
| Format | Four timed word slots per sentence (2 s + 2 s + 2 s + 3 s display) |
| Word timestamps | Available (begin/end per word or word group) |
| Split | **370 / 30 / 100 sentences** (train / val / test) |
| Metrics | PER **13%**, WER **14%** |
| Model | Single-layer GRU, 100 epochs, best val loss checkpoint |

Sentence template (timing from paper):

```
<weekday>_2s - <month>_2s - <date>_2s - <year>_3s
```

---

## Personalization / geometry (Appendix A.2)

- Eigenbasis **Q** is **subject-specific** — cross-subject shift = **change of basis** in ℝ<sup>|𝒱|</sup>.
- Motivates per-user calibration in deployment (aligned with Gowda 2024 geometry paper).

---

## Comparison to Gaddy

Authors report WER **73%** without EMG–audio alignment vs Gaddy ~42–68% with alignment — **not directly comparable** (different targets, splits, alignment assumptions).

---

## See also

- [../methods/01-spd-features-pipeline.md](../methods/01-spd-features-pipeline.md)
- [../datasets/01-small-vocab.md](../datasets/01-small-vocab.md)
- [../openalterego/01-gap-analysis.md](../openalterego/01-gap-analysis.md)
