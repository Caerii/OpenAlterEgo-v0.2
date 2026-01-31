# Background: AlterEgo-style silent speech interfaces (public info)

This doc is a *working notes* summary of what’s described in public papers (and a relevant patent),
so we can build an open alternative without guessing.

## Key public sources we leaned on

- Kapur, Kapur, Maes (2018) — “AlterEgo: A Personalized Wearable Silent Speech Interface” (IUI 2018).
- Kapur et al. (2020) — “Non-Invasive Silent Speech Recognition in Dysphonic Multiple Sclerosis” (PMLR/ML4H).
- US20190074012A1 / US10878818B2 — “Methods and Apparatus for Silent Speech Interface” (patent; useful for understanding system blocks + risks).

## Paper → engineering takeaways (condensed)

### 2018 IUI paper (core “AlterEgo” prototype)
- **Channels / electrode target areas:** 7 channels sourced from laryngeal region, hyoid region, levator anguli oris,
  orbicularis oris, platysma, anterior belly of digastric, mentum.
- **Electrodes:** gold-plated silver electrodes + Ten20 conductive paste (reported as better quality) or passive dry Ag/AgCl.
- **Reference electrode:** wrist or earlobe.
- **Sampling / gain:** 250 Hz sampling, 24× gain (OpenBCI / TI front-end mentioned).
- **Preprocessing:** bandpass 1.3–50 Hz (4th order IIR Butterworth), 60 Hz notch, ICA, rectification, normalize 0–1.
- **Transport:** BLE to mobile device; mobile relays to server hosting recognition.
- **Features + model:** MFCC-style features (25 ms window, 10 ms hop, mel filterbank, DCT), then 1D CNN.
  CNN described as repeated conv+pool blocks (400 filters, kernel 3, stride 1), global max pool, FC(200), sigmoid output.
- **Wearable form:** band around back of head, photopolymer resin + brass rod, modular brass electrode supports.

### 2020 ML4H paper (clinical MS patients, dysphonia)
- **Electrodes:** 8 signal electrodes (4 face + 4 neck) and reference/bias electrodes on each earlobe.
- **Sampling:** 24-bit ADC, 250 Hz sampling reported.
- **Preprocessing:** normalize initial value, high-pass (>0.5 Hz), notch at 60 Hz and harmonics,
  heartbeat removal via Ricker wavelet subtraction, band-pass 0.5–8 Hz (based on their analysis).
- **Task:** 15 sentence classes, each repeated 10× (per subject), personalized models.
- **Model:** CNN over fixed-length sequences (900 samples × 8 channels), Adam optimizer (1e-4), batch size 50,
  fc1 has 250 units, dropout 0.5 used (exact conv kernel/filter sizes were in a figure).

## What this repo does (scope)

We’re building an **open modular platform**:
- Works with 7–8 channels at ~250 Hz (but can do higher rates).
- Supports both “classic AlterEgo-like” preprocessing and newer variations.
- Provides baseline ML models + a websocket API so you can pipe decoded tokens into Unity / XR.

