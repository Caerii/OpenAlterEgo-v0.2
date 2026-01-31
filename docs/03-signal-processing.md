# Signal processing

This project supports two “families” of preprocessing, because public work reports
different filter bands depending on the experiment.

---

## Pipeline A: “Classic AlterEgo” (2018-ish)

Typical steps:
1. Bandpass (about 1.3–50 Hz), IIR Butterworth
2. Notch at 60 Hz
3. ICA (optional) to reduce motion artifacts
4. Rectify (abs)
5. Normalize per-channel (0–1 or z-score)
6. Segment windows based on markers

---

## Pipeline B: “Clinical / low-frequency emphasis” (2020-ish)

Typical steps:
1. Normalize to initial value
2. High-pass (>0.5 Hz)
3. Notch at 60 Hz and harmonics
4. Heartbeat artifact suppression (wavelet subtraction)
5. Bandpass 0.5–8 Hz (as reported)

---

## Practical default in this repo

We default to:
- high-pass ~0.5–1 Hz
- low-pass 40–50 Hz
- notch 60 Hz (or 50 Hz depending on mains)
- optional ICA (offline)

Because it’s robust and lightweight.

---

## Feature representations

Two options:

### 1) Learn from “raw-ish” time series (recommended first)
- Use normalized channels as input into a 1D CNN or a small transformer.

### 2) MFCC-like features (paper-faithful)
If you want to mimic audio-ish envelopes:
- frame into 25 ms windows with 10 ms hop
- compute periodogram / PSD
- apply mel filterbank
- log + DCT → MFCC-like coefficients
- then classify with CNN

