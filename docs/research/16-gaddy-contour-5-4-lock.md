# Locked 5-4 activity and slow-contour evaluation

This protocol is fixed before extracting voiced parallel session 5-4. The
archive listing shows 238 trial IDs in that session; its trial metadata,
signals and labels remain unopened. The first 100 matched trials are the
reserved test subset. The publisher archive must pass MD5
`7f97d2182b896652999b1b2d0c69fd7b` before extraction outside Git.

## Development and model choice

The [development probe](../../software/python/scripts/gaddy_contour_activity_probe.py)
uses only sessions 5-5, 5-6, 5-8, 5-9, 5-10 and 5-11, all previously
examined. Rotations held out 5-10 and 5-11 in turn. The outcome metric is
the MAE of 200 ms changes in a causally smoothed log10 audio-RMS contour,
restricted to uninterrupted audio-defined speech-active frames. On 5-10,
full soft gate scored 0.273, top-two sparse latent soft gate 0.281, global
speech-trained head 0.309 and zero change 0.353. On 5-11 the same sequence
was 0.312, 0.307, 0.324 and 0.361. The [5-10 result](results/gaddy-contour-activity-dev-5-10.json)
and [5-11 result](results/gaddy-contour-activity-dev-5-11.json) preserve
other metrics. The sparse model does not consistently win; it remains a
predeclared challenger.

## Frozen recipe and decisions

- Pool eligible first-100 trials from voiced sessions 5-5, 5-6, 5-8, 5-9,
  5-10 and 5-11 for training. Exclude `sentence_index < 0`. Evaluate all
  eligible trials in the first 100 of 5-4. No 5-4 feature or label
  calibration, parameter choice or model selection is allowed.
- Filter each EMG trial forward-only, then compute eight log RMS and eight
  log waveform-length features from 100 ms windows at 50 ms steps. Stack
  current, 50, 100 and 200 ms past frames within each utterance. The
  acoustic target is log10 RMS in a 40 ms window, averaged over the current
  and four past frames (250 ms causal smoothing). The EMG window is centered
  on the acoustic frame, so emission occurs approximately 50 ms after its
  center; this is a delayed forward-only predictor, not zero-latency output.
- Audio-defined activity is raw log10 RMS above -2.0. A gradient-boosted
  classifier predicts it from full EMG history. Train a separate regressor
  on activity-positive training frames to predict smoothed energy. Combine
  its output with the mean inactive-training target by the past-five-frame
  mean predicted activity probability. All gradient boosters use 120
  iterations, 15 leaves, learning rate 0.06, L2 10 and seed 42.
- Primary **slow-contour decision**: the full soft-gated model's 200 ms
  delta MAE must beat both zero change and the global-activity-only
  speech-trained head. Both paired utterance-bootstrap 95% intervals for
  control-minus-primary advantage must be entirely above zero. Report
  classifier average precision, ROC AUC and F1, full-frame and oracle
  active-frame energy MAE, and contour direction accuracy as secondary.
- Sparse-prior decision: eight nonnegative EMG feature patterns are learned
  without audio labels using the fixed MiniBatchNMF configuration in the
  probe. Keep only the two largest codes per frame. The sparse soft-gated
  model must beat the full soft-gated model with a positive paired
  utterance-bootstrap interval to claim that this **specific** sparse prior
  improves predictive decisions. Also report the dense latent variant.
  No claim about identified muscles or causal electrode interventions can
  follow from a fixed-montage session comparison.

Bootstrap uses 5,000 whole-utterance draws, seed 42. Evaluation masks use
recorded audio only after prediction; they are not deployable silent-speech
labels. Vocalization can alter EMG and sensor pickup. No result here can
establish silent intended prosody, emotion or personal-voice reconstruction.
Once opened, 5-4 becomes development evidence; further tuning needs a new
dataset or genuinely unused sessions.
