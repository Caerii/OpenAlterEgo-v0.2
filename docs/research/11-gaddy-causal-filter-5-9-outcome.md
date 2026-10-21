# Locked 5-9 causal-filter outcome

The [protocol](10-gaddy-causal-filter-session-lock.md) was pushed in commit
`d97c130b9e468603884ea1d785a321706a6ecaa8` before 5-9 extraction.
The archive MD5 matched the publisher value. Among IDs 0-99, 79 trials were
non-reference: 10 supplied unlabeled calibration and 69 supplied evaluation.
The 5-5 training subset contained 83 non-reference trials. Book/sentence keys
and normalized text strings had **zero overlap** between the training and
evaluated subsets. These are different trials from the same speaker.

| Target and preprocessing | Past-only calibrated EMG MAE | Training-mean prior | Matched shuffled-training control | Decision |
| --- | ---: | ---: | ---: | --- |
| All-frame log10 audio RMS, forward-only filter **(locked primary)** | **0.590** | 0.816 | 0.807 | Passes both prespecified comparisons |
| Voiced-frame log10 audio RMS, forward-only filter | 0.548 | 0.818 | 0.831 | Secondary; voiced mask uses acoustic truth |
| Voiced-frame F0 Hz, forward-only filter | 72.77 | 89.46 | 89.96 | Secondary; crude pitch label and acoustic voiced mask |
| All-frame log10 audio RMS, published-style offline filter | 0.574 | 0.816 | 0.818 | Secondary, uses forward-backward filtering and neighboring trials |

The primary absolute MAE reduction is 0.226 log10-RMS units, or 27.7% versus
the prior. The **unnormalized** forward-filtered model scored 0.560, lower
than the 0.590 calibrated score; hence the result supports signal transfer,
not a claim that ten-trial calibration is necessary. The full-session
transductive normalization scored 0.592. The secondary offline condition is
close to the forward-only result and cannot justify future-data access.

[Primary machine-readable result](results/gaddy-prosody-5-9-causal-filter-locked.json)
and [offline comparison](results/gaddy-prosody-5-9-official-offline-secondary.json)
record frame counts, trial IDs, session provenance, scores, and controls.
The primary analysis uses 9,010 highly correlated frames; no utterance-level
confidence interval was prespecified, so the point estimate must not be read
as a precise population effect. The calibration prefix and model were chosen
with earlier session evidence; this is one locked new-session check, not an
independent multi-person replication.

The result is encouraging for a **vocalized** EMG acoustic-energy cue. It does
not establish silent intended loudness or emotion. Actual phonation can
contribute muscle activity and acoustic/vibratory artifacts to the electrodes;
those sources disappear or change during silence. The 100 ms feature window
is available only at its end; audio target alignment has not been reconciled
exactly with Gaddy's published features. The energy error remains large for
fine expression control, and the simple F0 target is inaccurate enough that
the F0 comparison is not evidence of natural pitch reconstruction.

Next: prespecify utterance-level uncertainty and within-utterance temporal
negative controls, then test them on untouched 5-10. Separately evaluate
silent-to-voiced transfer using paired prompts, without equating a parallel
voiced recording's F0 with the silent wearer's intended F0.
