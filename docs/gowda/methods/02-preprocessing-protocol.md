# Preprocessing Protocol (Paper)

From [arXiv:2502.05762v2](https://arxiv.org/html/2502.05762v2) Appendix B.

---

## Signal chain

| Step | Operation | Parameters |
|------|-----------|------------|
| 1 | **Reference subtraction** | Channel 32 (right earlobe ref) subtracted from all EMG channels |
| 2 | **Bandpass** | 3rd-order Butterworth **80–1000 Hz** |
| 3 | **Segmentation** | Sentence boundaries from **mouse-click timestamps** |
| 4 | **Normalization** | **Z-score per channel along time** within each sentence segment |
| 5 | **Feature build** | Construct ℰ(τ) and σ(τ) per sliding window |

---

## Acquisition parameters

| Parameter | Value |
|-----------|-------|
| fs | **5000 Hz** |
| Electrodes | 31 monopolar + gnd + ref |
| Amplifier | actiCHamp Plus (Brain Vision) |
| Gel | SuperVisc (Easycap) |

---

## OpenAlterEgo `emg_mode` comparison

| Setting | Bandpass | Notes |
|---------|----------|-------|
| Paper | **80–1000 Hz** | 3rd-order Butterworth |
| `standard` | 1–50 Hz | AlterEgo heritage |
| `wide` | **20–450 Hz** | Literature EMG (not identical to paper) |
| Import z-score | Per-trial per channel | Matches emg2speech notebook ✓ |

**Recommendation for closer paper match:** add `gowda` or `clinical-wide` mode at 80–1000 Hz when fs=5000; use per-event preprocess reset.

---

## Segmentation differences

| Source | Segmentation |
|--------|--------------|
| Paper large-vocab | Whole **sentence** (click to click) |
| Paper small-vocab | **Per-word timestamps** (ideal) |
| emg2speech notebook | Fixed **10k/10k/10k/15k** sample splits |
| OpenAlterEgo import | emg2speech splits + `events.csv` contiguous indices |

Per-word **mouse timestamps** in OSF may be richer than fixed splits; future importer could use sidecar timestamp files if released.
