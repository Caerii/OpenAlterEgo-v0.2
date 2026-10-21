# Locked temporal audit: Gaddy session 5-10

Recorded 2026-10-21, after inspecting the 5-9 outcome and before extracting,
scoring, or viewing trial contents from session 5-10. Archive directory names
were inspected; 5-10 is the next numbered voiced parallel session after 5-9.
The [5-9 result](11-gaddy-causal-filter-5-9-outcome.md) motivated this stricter
test. It is a new prespecified check, not an independent replication of the
5-9 protocol.

## Fixed procedure

- Use the publisher-MD5-verified Gaddy archive. Train on valid IDs 0-99 of
  voiced parallel session 5-5. Extract valid IDs 0-99 of session 5-10. Its
  earliest ten valid trials provide unlabeled past-only median/IQR calibration;
  every subsequent valid trial is evaluated. If files are missing, report it
  and stop instead of substituting another session or count.
- Run `gaddy_prosody_probe.py --calibration-trials 10
  --emg-preprocessing causal-filter --temporal-audit` at the version committed
  with this lock. Keep the 100 ms EMG window, 50 ms hop, forward-only filters,
  fixed features, `StandardScaler`, and ridge alpha 100. Train and test on
  **all-frame log10 audio RMS** without acoustic oracle masking.
- For each evaluated utterance, record the model, training-mean, shuffled-
  training, and 20-frame circularly shifted EMG MAE. The shift is fixed at
  one second and leaves each utterance's EMG feature distribution intact while
  disrupting its frame timing. This control is evaluated retrospectively.
- Resample whole utterances as paired units, 5,000 bootstrap draws with seed
  42. For each control, compute the mean utterance-level MAE difference
  `(control - aligned EMG)` and its percentile 95% interval.
- **Primary success** requires the 95% interval lower endpoint to be strictly
  positive for **all three** controls: training mean, shuffled training, and
  one-second shift. Also report frame-weighted MAE and R², but do not use
  their values to revise the decision.
- Record train/test book-sentence and normalized-text overlap, the valid
  trial count, secondary voiced-frame energy and F0, and all failures.

The corpus is one speaker's vocalized EMG. A positive result establishes a
session-transferable, time-sensitive association with recorded acoustic
energy under these settings. It cannot establish silent intended prosody;
phonation-related EMG or mechanical pickup remains a competing explanation.
The arbitrary one-second lag does not exhaust all temporal confounds, and
the baseline is a small ridge probe, not a production decoder.
