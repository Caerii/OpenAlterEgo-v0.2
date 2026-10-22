# Frozen seven-person new-session calibration result

**Status:** completed method check on previously unused Hyser subjects
06–08 and 17–20. [Hyser](https://physionet.org/content/hd-semg/2.0.0/)
records forearm gestures, not facial silent speech. The
[development result](35-hyser-labeled-calibration-development.md) and
[frozen protocol](36-hyser-fewshot-holdout-protocol.md) preceded this cohort.
The protocol and scoring code were pushed in commit `1eb9096`. Day-1
alpha/layout [choices](results/hyser-fewshot-holdout-v1-selection.json) were
pushed in commit `b4c9ff5` before any of these day-2 recordings were
downloaded. The selection record's canonical SHA-256 is
`5dc8909099af2584d1c3f038f62ae3447f351f3eebc957be0a90a67188cf0425`.
The [score record](results/hyser-fewshot-holdout-v1-score.json) binds that
hash to the publisher-derived source hashes and exact support/query counts.

Both models use the same fixed eight recorded contacts and day-1 ridge
alpha, selected by day-1 CV. All seven selected alpha 1. For the primary
comparison, the first recorded repetition of each available day-2 gesture
provides a labeled support example with per-class weight 1; remaining
repetitions form the query set. The baseline ignores support and is scored
on the **same** query trials.

| Subject | Support / query trials | No support | One-shot support | Paired gain |
| --- | ---: | ---: | ---: | ---: |
| 06 | 34 / 167 | 0.576 | 0.782 | +0.206 |
| 07 | 34 / 170 | 0.553 | 0.759 | +0.206 |
| 08 | 34 / 169 | 0.551 | 0.665 | +0.113 |
| 17 | 34 / 168 | 0.294 | 0.690 | +0.396 |
| 18 | 34 / 169 | 0.403 | 0.697 | +0.294 |
| 19 | 33 / 164 | 0.739 | 0.861 | +0.121 |
| 20 | 34 / 170 | 0.165 | 0.565 | +0.400 |
| **Mean of subjects** | | **0.469** | **0.717** | **+0.248** |

The predeclared decision rule passes: the mean gain is positive, all seven
people improve, and none has a loss. Subject 19's second session omits
gesture class 11, so its support set has 33 prompts rather than 34. These
are paired participant-level outcomes, not 1,177 independent observations.
As a descriptive small-sample interval, a conventional t interval for the
seven paired gains is +0.138 to +0.358; its coverage depends on assumptions
that this selected public cohort cannot establish.

The frozen secondary comparison used two labeled repetitions per class at
per-class weight 4. On its own query subset, mean accuracy rose from 0.454
without support to 0.785 with support: a +0.331 mean paired gain. All seven
improved. This uses 66–68 prompted responses, compared with 33–34 for the
primary one-shot rule, and is not directly comparable to the one-shot query
accuracy because its query trials differ.

## What this does and does not establish

This establishes that a **small labeled sample from the same new session**
can substantially repair a fixed-layout decoder on Hyser gestures under
this specific episodic split. It shifts the research emphasis from trying to
guess a perfect permanent electrode layout toward efficient user calibration
and adaptation. It does not establish that the eight-site wearable design
is optimal, that contact physics are modeled correctly, or that silent
speech can be decoded with these accuracies.

The major limitation is recording order. Hyser groups all repetitions of a
gesture together. A support example for a later gesture was recorded after
query examples for earlier gestures. The split is disjoint by trial but
**not globally chronological**. Immediate within-block similarities may
inflate transfer relative to an actual workflow where all calibration
prompts are collected first and diverse commands are attempted later.
The method also requires labeled prompted gestures and uses only a
constrained 33–34-class task. One-shot calibration costs at least 33–34
seconds of signal time, excluding prompts, rests, donning, and fitting.

The next decisive data test should use a facial/neck EMG corpus with true
session and time ordering, and a protocol that physically separates the
calibration block from subsequent evaluation. The official
[CSL-EMG_Array corpus](https://www.uni-bremen.de/en/csl/research/silent-speech-communication/csl-emg-array-corpus)
is a candidate but needs access and inspection. A future physical wearable
study must additionally measure reseating, impedance, and movement.
