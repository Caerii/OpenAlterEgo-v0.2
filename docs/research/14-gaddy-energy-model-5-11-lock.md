# Locked 5-11 energy-model evaluation

This protocol is fixed before extracting or evaluating voiced parallel session
5-11. It investigates energy prediction from **vocalized** EMG, not intended
expression during silent articulation. The publisher archive must pass its
published MD5 before a bounded first-100-trial extraction outside Git.

## Development observations and decision

The [comparison script](../../software/python/scripts/gaddy_energy_model_compare.py)
was designed on already examined sessions. Four-session pooled training with
5-10 held out for development yielded all-frame log10-RMS MAE 0.557 for the
current ridge, 0.484 for a causal past-window ridge, 0.400 for a two-summary
global-activity gradient booster, and 0.340 for the full 16-feature
past-window gradient booster. On an independent development rotation holding
out 5-9, corresponding values were 0.556, 0.469, 0.401, and 0.349. The
full-model advantage over the global-activity baseline is useful evidence
that channel-specific patterns matter. A one-second past feature shift scored
0.918 on 5-10 and 0.955 on 5-9. [Machine-readable results](results/gaddy-energy-model-compare-dev-5-10.json)
and [second rotation](results/gaddy-energy-model-compare-dev-5-9.json) retain
the exact frame counts and other metrics.

These results do **not** establish expressive contour tracking. On 5-10, the
full model's MAE for 50 ms energy changes inside continuously voiced segments
was 0.205, versus 0.192 for predicting no change; on 5-9, 0.189 versus
0.169. The audio-derived voiced-interior mask is an oracle evaluation
selection and is never fed to the predictor. Its simple autocorrelation pitch
detector can make mistakes. Vocalization itself may influence the electrodes
and cannot establish silent-speech performance.

## Frozen 5-11 recipe

- Extract only the first 100 matched EMG/audio/info trials from voiced
  parallel session 5-11, discard `sentence_index < 0`, and evaluate all
  remaining eligible trials. No 5-11 label or feature calibration or model
  selection is permitted.
- Pool eligible trials from the first 100 of voiced sessions 5-5, 5-6, 5-8,
  5-9 and 5-10 as training. The evaluation session must remain disjoint.
- Apply the per-trial forward-only 60 Hz harmonic notches and 2 Hz highpass;
  use 100 ms EMG windows at 50 ms steps and aligned 40 ms audio windows.
  Features are eight log RMS and eight log waveform-length channels. Stack
  current, 50, 100 and 200 ms past feature frames, repeating the first frame
  within an utterance at boundaries.
- Compare the fixed `HistGradientBoostingRegressor` (120 iterations, 15
  leaves, learning rate 0.06, L2 10, seed 42), current-frame ridge (alpha
  100 with standardized inputs), past-window ridge (same alpha), training
  mean, and identical gradient booster trained on two global activity
  summaries. Shift the full model's test features one second into the past
  as a temporal control. Do not optimize these choices on 5-11.
- Primary energy-engineering decision: full model all-frame log10-RMS MAE
  must beat both current ridge and global-activity booster, with the 95%
  paired utterance-bootstrap intervals for both control-minus-full-model
  advantages entirely positive. The aligned model must also beat its
  one-second shift. Report all-frame and oracle voiced-interior MAE.
- Separate expression decision: full model's 50 ms voiced-interior energy
  change MAE must beat the zero-change baseline. Failing this prohibits a
  claim that the model tracks fine prosodic contours, even if the primary
  energy decision passes.

Bootstrap uses 5,000 whole-utterance resamples with seed 42. This estimates
uncertainty across these utterances conditional on a single speaker and
session; it does not imply independent participants. The outputs stay within
the research result set; raw data remain outside Git. Further model changes
require another untouched evaluation session.
