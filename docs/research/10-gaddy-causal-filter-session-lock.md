# Locked cross-session signal check: Gaddy 5-9

Recorded 2026-10-21 23:29 UTC, before extracting or scoring any 5-9 trial.
The archive directory list includes sessions 5-4, 5-5, 5-6, 5-8, 5-9,
5-10 and 5-11. We choose **5-9**, the earliest session after the already
examined 5-8, by this temporal rule. Sessions 5-5, 5-6 and 5-8 have supplied
development outcomes. Do not substitute 5-10 or 5-11 after seeing 5-9 results.

## Fixed experiment

- Input is the MD5-verified [Gaddy archive](https://zenodo.org/records/4064409).
  Train on IDs 0-99 in voiced parallel session 5-5; evaluate IDs 0-99 of
  session 5-9. Exclude `sentence_index: -1`; if any required archive component
  is absent, report inability to run this locked test. Neither 5-9 audio nor
  EMG is inspected before this protocol is committed and pushed.
- Use `gaddy_prosody_probe.py --calibration-trials 10
  --emg-preprocessing causal-filter`. The first ten valid 5-9 trials supply
  unlabeled median/IQR calibration only. All later valid trials are evaluated.
  The 5-5 training set supplies model weights and its own normalizer. The
  ridge alpha is 100. The 100 ms EMG window and 50 ms step remain fixed.
- The forward-only notch/high-pass filters operate at 1000 Hz with state reset
  per trial. A centered 100 ms feature window becomes available at its end,
  at least 50 ms after its nominal center. Thus this checks causal signal
  availability with bounded window delay, not end-to-end interaction latency.
- **Primary endpoint:** all-frame log audio RMS MAE. A positive result requires
  calibration-only EMG MAE to be strictly below **both** the training-mean
  prior MAE and the calibration-domain shuffled-training EMG MAE. Do not
  change the calibration prefix or threshold after 5-9 is opened.
- Secondary: voiced-frame energy and F0 MAE (their voiced mask uses acoustic
  ground truth, so they are oracle-stratified diagnostics); unnormalized,
  full-test-session normalized, and `official-offline` preprocessing outputs;
  train/test book-sentence overlap and text overlap. Report all outcomes.

## Claim boundaries

This tests one speaker's *vocalized* EMG under recording-session shift.
Prompt overlap must be audited after extraction before any unseen-text claim.
It cannot validate silent intended prosody, voice identity,
affect, an actual wearable, or a generative voice channel. The forward-only
filter differs from Gaddy's published forward-backward filter, and the probe's
100 ms frame alignment is not a reproduction of the published audio-feature
alignment. There has been no hyperparameter search on a fresh validation
cohort. Passing this single threshold would only justify a larger study.
