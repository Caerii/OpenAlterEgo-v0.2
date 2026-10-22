# Locked Gaddy unseen-trial-range check

This protocol is committed before extracting or scoring trial IDs 100–199 of
parallel vocalized sessions **5-4** and **5-8**. The [earlier result](65-gaddy-sparse-energy-session-search.md)
chose the six-channel mean-session program after inspecting trial IDs 0–99 in
other sessions. Both query sessions have also appeared in earlier research, so
this is a previously unscored **trial-range** check, not a new-person or
new-session test. Do not use its results to revise the locked choice and then
describe the same ranges as a holdout.

## Frozen comparison

- Source: published [Gaddy archive](https://zenodo.org/records/4064409),
  verified against MD5 `7f97d2182b896652999b1b2d0c69fd7b`.
- Fit: parallel 5-5, trial IDs 0–99, same cached Rust energy features and
  batched ridge scorer as the earlier assay. Alpha 100; four past-only lags
  `(0, 1, 2, 4)` of two features per recorded EMG channel. Training-only
  feature standardization and unpenalized intercept.
- Query: parallel 5-4 and 5-8, trial IDs 100–199 inclusive. In each query
  range, the first ten **valid** utterances provide unlabeled median/IQR
  normalization. Score log acoustic energy MAE on all remaining valid
  utterances. Never use their acoustic targets or later EMG frames to fit,
  normalize, or select a model.
- Frozen choices: multi-session mean `(0,1,2,3,4,6)`; earlier-session
  `(0,1,3,4,6,7)`; index-fixed `(0,1,2,4,5,7)`; all eight recorded channels.
  No new channel search on these ranges before reporting the four frozen
  programs. Channel IDs refer to recorded channels, not mapped skin contacts.
- Primary comparison: the unweighted mean of the two query-session frame
  MAEs, multi-session six versus index-fixed six. Report each session too.
  Secondary comparisons are versus the earlier-session six and all eight.
  Record per-utterance errors and valid trial IDs. A smaller mean alone is
  exploratory evidence; check whether the direction holds in **both** sessions.

No speech recognition, silent-prosody, contact placement, or anatomical
generalization claim follows from this vocalized one-speaker energy task.
Even a favorable result must be followed by new sessions, speakers, and
physical montage interventions for the proposed wearable objective.
