# Prosody third-session check: locked before extraction

Recorded 2026-10-21 before extracting or scoring `voiced_parallel_data/5-8`.
The first within-session probe and 5-5 → 5-6 cross-session failure are already
development evidence in [the expressive-voice plan](06-expressive-silent-voice.md).
This check tests a single revised rule, with no further feature, target,
threshold, or hyperparameter selection on 5-8.

## Fixed protocol

- Source: MD5-verified Gaddy archive from [Zenodo 4064409](https://zenodo.org/records/4064409).
- Train: IDs 0–99 of `emg_data/voiced_parallel_data/5-5/`, excluding
  `sentence_index: -1`. Test: IDs 0–99 of
  `emg_data/voiced_parallel_data/5-8/`, with the same exclusion. No test
  acoustic target enters fitting or model selection. Parallel prompts may
  repeat across sessions; this is a recording-shift check, not unseen text.
- Model: `scripts/gaddy_prosody_probe.py` as committed with this protocol;
  100 ms eight-channel EMG log RMS and log waveform length, 40 ms audio
  target, 50 ms step, `StandardScaler` plus `Ridge(alpha=100)`.
- Main intervention: independently center/scale all unlabeled training and
  test-session EMG feature frames by their median and IQR, floor 0.1. This
  uses the entire unlabeled test session offline and is not a live-stream
  calibration claim.
- Primary metric: voiced-frame log-audio-RMS MAE. Success means the normalized
  EMG MAE is **strictly less** than the training-mean prior MAE on 5-8.
- Secondary metrics: raw EMG, shuffled-training EMG, all-frame energy, and
  autocorrelation F0 MAE and R². Report whether F0 also beats the prior;
  it does not gate the primary energy hypothesis. Preserve negative outcomes.

Even success would be narrow: one speaker, voiced speech, repeated prompts,
and an exploratory normalization rule chosen on 5-6. Silent prosody, emotion,
voice identity, and a no-mouth-motion interaction remain separate claims.
