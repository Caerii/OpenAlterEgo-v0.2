# Where the fixed-layout recognition performance is lost

This retrospective development study uses SilentWear S01–S03 only. Session 3
has been examined in prior work. The results guide the next experiment; they
are not an independent accuracy estimate for a wearable, a physical electrode
intervention, or unrestricted speech. The nine-class outcome is isolated,
centered command/rest events, not streaming recognition.

## Does a wider HDC vector or a local search plateau help?

The [fixed-layout representation controls](27-fixed-layout-representation-controls.md)
showed that the 512–8192-dimensional Torchhd projection-centroid and a small
raw-window CNN trailed feature logistic regression. The limitation could be
the six-channel budget, day-to-day signal drift, or noisy program selection.

We tested a parameter-free one-swap prior: for each six-channel program,
average its bidirectional ridge-proxy score with all 48 programs formed by
replacing exactly one recorded channel. Smoothing is done separately for each
normalization and each fit/validation direction, then the lower directional
score selects the program. Both selected programs use the same C=1 logistic
decoder fit on sessions 1 and 2 and scored on session 3. Choices were written
to exclusive local records before session 3 was loaded in this run. This is
still retrospective because earlier work exposed session 3.

| Person | Raw bidirectional choice | One-swap smoothed choice |
| --- | ---: | ---: |
| S01 | 0.2659 | 0.2622 |
| S02 | 0.2867 | 0.2835 |
| S03 | 0.3170 | 0.3152 |
| **Mean** | **0.2899** | **0.2869** |

The smoothing rule moved every choice but slightly reduced every future-day
score. It is inexpensive compared with the 10–14-second per-person proxy
screen, yet offers no measured benefit here. A one-index substitution also
has no anatomical distance meaning; this is a combinatorial regularizer, not
placement-perturbation physics. [Full result and frozen choices](results/silentwear-neighborhood-prior-v2.json).

## How much do all observed channels help?

With the same recording-robust features and logistic classifier fit on both
early sessions, the [14-channel ceiling diagnostic](results/silentwear-channel-ceiling-v1.json)
compares the fixed six-channel index subset with all 14 *recorded* channels.

| Person | Fixed six, future day | All 14, future day |
| --- | ---: | ---: |
| S01 | 0.3558 | 0.3275 |
| S02 | 0.2588 | 0.3086 |
| S03 | 0.2783 | 0.3358 |
| **Mean** | **0.2976** | **0.3240** |

All channels help S02 and S03, but hurt S01. This is a finite observed-channel
reference, not a ceiling on physiology or an estimate at unrecorded sites.

## Same fitted model, current versus future day

To isolate day shift more directly, one classifier is fit on all session-1
batches plus session-2 batches 1–4. It is evaluated on session-2 batch 5 and
all five session-3 batches. The train set has 2,889 events; the near and future
sets have 321 and 1,605 events respectively. [Full split and scores](results/silentwear-session-gap-v1.json).

| Channels | Near-session mean | Future-session mean | Absolute drop |
| --- | ---: | ---: | ---: |
| Fixed six | 0.3710 | 0.2830 | 0.0880 |
| All 14 | 0.4464 | 0.3080 | 0.1384 |

The near-session advantage from eight extra channels is 0.0754; on the future
day it is only 0.0250. The comparison uses the same fitted model, but the
test sets differ in size and recording day, and there are only three people.
It strongly prioritizes *session transfer* as a hypothesis, without proving
which mechanism causes the gap. Candidate causes include contact, placement,
anatomy, task performance, and acquisition drift. The available files do not
identify those causes independently.

## A first current-day calibration test

We then fixed a calibration pool to session-3 batch 1 and evaluated only
batches 2–5. The [prototype curve](results/silentwear-calibration-curve-v1.json)
uses nested random samples of 0, 1, 5, or 20 events per class from batch 1,
with five prior examples' weight on each old-session centroid. With all 14
channels, the mean balanced accuracy was 0.2585 with zero calibration,
0.2581 with one, 0.2813 with five, and 0.2948 with twenty examples per class.
The frozen logistic model on the same later batches scored 0.3202. This
particular Euclidean prototype update did not recover the lost performance.

As a more expensive calibration reference, a
[fresh logistic model](results/silentwear-session-local-v1.json) trained only
on all 321 labeled events in session-3 batch 1 scored 0.5196, 0.2764, and
0.3255 on the later batches with all 14 channels (mean 0.3738). The frozen
earlier-session logistic model scored 0.3206, 0.3140, and 0.3261 on those
same later batches (mean 0.3202). **S01 accounts for the gain; S02 and S03 do
not benefit.** This is a sizeable explicit user calibration burden, and its
uneven effect rules out a general claim that short calibration solves drift.
It does show that some of S01's day shift is learnable from same-day labels.

## Detection is easier than command identity

The [two-stage diagnostic](results/silentwear-hierarchical-v1.json) fits a
binary rest/command gate and a separate eight-command classifier on sessions
1+2. The threshold is fixed at 0.5. With all 14 recorded channels on session
3, binary gate balanced accuracy is 0.953, 0.829, and 0.913 for S01–S03.
Conditional eight-command balanced accuracy, evaluated only on true command
events, is 0.274, 0.300, and 0.285. The full two-stage nine-class score is
0.333, 0.324, and 0.334, compared with 0.327, 0.309, and 0.336 for the flat
logistic model. The mean gain is only 0.006.

Thus this fixed feature set usually detects *that a command is happening*
better than it identifies *which command*. A bigger rest detector alone is
unlikely to unlock a large accuracy gain. The next representation experiment
should target articulatory distinctions among commands, especially those with
near-zero per-class recall in the
[14-channel confusion matrices](results/silentwear-channel-ceiling-v1.json).

## Next experiment with the highest expected value

1. Audit S02/S03 command-pair confusions, signal quality, and batch-to-batch
   feature shifts. The frozen classifier and 321-example same-day fit both
   remain poor for them. Test richer temporal/spatial command features with
   a frozen protocol and a larger independent dataset. A regularized
   calibration layer can then be evaluated at a fixed utterance budget; the
   first prototype rule failed this test.
2. Extend the evaluation to an independently sourced corpus. The
   [Gaddy facial-EMG corpus](https://zenodo.org/records/4064409) has multiple
   recording sessions and eight channels; it can test transfer methods but
   cannot validate 14-channel placement search. The
   [CSL-EMG_Array corpus](https://www.uni-bremen.de/en/csl/research/silent-speech-communication/csl-emg-array-corpus)
   has cheek and chin arrays, eight speakers and twelve sessions; its 8.8 GB
   download currently requires an emailed link. Its
   [paper](https://www.isca-archive.org/interspeech_2020/diener20b_interspeech.html)
   describes spatially resolved facial EMG, though only four sessions use the
   silent test condition. Preserve each corpus's actual geometry, task and
   session boundaries instead of pooling labels as if they were equivalent.
3. For wearable-design search, collect or obtain **paired measurements under
   known electrode interventions**. Until such data exist, report only subset
   robustness on observed channels. A neural operator trained on an anatomical
   solver can accelerate counterfactual predictions, but its layout ranking
   still needs measurement-based validation.

The highest-leverage immediate work is to explain the person-specific failure
and establish a new-corpus gate, with the current 14-channel logistic model as
an observed-information reference. Greater search throughput is useful once
the score being optimized predicts performance on a new day.
