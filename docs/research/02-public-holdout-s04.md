# Locked public-data holdout: SilentWear S04

Registered locally before S04 download or inspection on 2026-10-21. This is a
public-data validation plan, not an independent cryptographic preregistration
or a physical wearable result. The code and development result are on the
`research/physics-pilot` branch. The complete S01–S03 development output is
`results/silentwear-development-cohort.json`.

## Frozen procedure

- Import the publisher's S04 silent files at dataset revision
  `9fd2d8f5f8e51333b5f3314d878163710be2b3b2`, five batches per session,
  three sessions. Verify each against its publisher SHA-256.
- Run `scripts/silentwear_cohort.py` with **S04 alone**. Use the current fixed
  five six-channel subsets, centered 400-sample windows, log RMS and waveform
  length, balanced logistic regression, and 20 synthetic feature-drift draws
  at sigma 0.35. No architecture, features, hyperparameters, or candidate
  subsets may be chosen using S04 outcomes.
- Primary selection rule: label-free robust normalization within each batch;
  train on session 1; choose the subset with highest balanced accuracy on
  session 2; report its balanced accuracy and decision regret on session 3.
  Session 3 labels are never used for normalization or model fitting.
- Comparators: unnormalized session-2 selection and the fixed `low_index`
  subset under the same normalized model. Report all five subsets and
  pairwise ranking accuracy so a failed choice remains visible.

## Predictions and interpretation

The exploratory S01–S03 results motivate a prediction that the normalized
selector's S04 pairwise ranking accuracy exceeds 0.5 and its session-3 regret
is at most 0.03 balanced-accuracy points. These thresholds are hypotheses,
not established capabilities. A single fourth participant provides a
diagnostic replication, not a population estimate. If either threshold fails,
retain the result and revise on a *new* corpus or previously untouched split.

Even a pass would establish only ranking of subsets of SilentWear's existing
recorded electrodes under this fixed protocol. It would not validate moving an
electrode, a personalized anatomical model, or OpenAlterEgo hardware.
