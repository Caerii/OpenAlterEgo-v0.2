# A small support set predicts part of facial EMG session shift

The [eight-session channel search](75-emg-uka-nested-session-rule-audit.md)
showed that development scores do not reliably select the best recorded
channels on a new session. We tested a narrower modeling question on the
same [EMG-UKA trial corpus](https://www.kaggle.com/datasets/xabierdezuazo/emguka-trial-corpus):
can a simple nuisance shift estimated from a few target-session prompts
predict the **remaining EMG features**? This is a retrospective,
speaker-008, audible-prompt assay, not an accuracy improvement claim.

The [runner](../../software/python/scripts/emg_uka_session_drift.py)
uses the previously defined six-channel, 60-dimensional features: five
relative-time bins of log RMS and log waveform length per channel. For
each target session 002–008, it makes a template for each of the ten
repeated prompts from **earlier numbered sessions only**. It estimates
an additive difference from the 2, 4, or 6 lowest-ID target utterances
and scores absolute feature error on the remaining utterances. Models
range from no shift, one global shift, six channel shifts, 12
channel-by-feature-type shifts, to 60 unconstrained shifts. Two controls
estimate global or channel shifts **without prompt labels** by subtracting
the earlier-session population template instead of matched prompt templates.
Source content digests, selected IDs, every target result, and 10,000-draw
paired session bootstrap intervals are in the [result record](results/emg-uka-session-drift-v1.json).

| Target support | No shift MAE | Global shift | Six channel shifts | Unlabeled six channel shifts | Full 60-feature shift |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 prompts | 0.2100 | 0.1722 | **0.1667** | 0.1735 | 0.1803 |
| 4 prompts | 0.2014 | 0.1605 | **0.1556** | 0.1563 | 0.1666 |
| 6 prompts | 0.1952 | 0.1650 | 0.1594 | 0.1614 | **0.1553** |

Units are mean absolute error in log ADC-derived features, averaged over
seven target sessions; different support sizes leave different evaluation
utterances. With two support prompts, the six-channel shift improves six
of seven sessions and lowers mean error by **0.0433**, about **21%** of
the no-shift error. The paired-session bootstrap interval for that
difference is **[0.0106, 0.0769]**. Session 003 is a clear failure:
its target error rises from about 0.16 to 0.20 after this correction.
The full 60-feature shift is worse than the six-parameter shift when
support is only two prompts, consistent with overfitting a small support.

The evidence for **channel-specific** structure is weaker than the evidence
for a broad session shift. With two support prompts, six channels improve
over one global shift by just 0.0055 MAE; a seven-session paired bootstrap
interval is approximately **[−0.0025, 0.0164]**. The unlabeled channel
control is close to the label-matched one, especially at four prompts.
This suggests a coherent change in observed amplitude/shape across a
session that may be estimated without labels. It does **not** identify
whether contact impedance, electrode placement, muscle recruitment,
behavior, or amplifier state caused that change.

The corpus documentation explicitly orders **sessions** by recording, but
does not establish that utterance IDs encode within-session acquisition
time. Thus the lower-ID support is disjoint from evaluation but cannot be
called verified past-only online calibration. The same ten prompts repeat
across days, all eight sessions have already been studied, and seven
target sessions are too few for a population estimate. The improvement
is useful as a measured *session-nuisance modeling target*. It should
inform a persistent session-shift stress scenario in the simulation
harness; no claim about a manufactured OpenAlterEgo electrode or silent
speech decoder follows from this assay.
