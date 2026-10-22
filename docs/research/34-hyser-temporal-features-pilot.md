# Temporal features and a generalized dense-program scorer

**Status:** development result on the 13 Hyser subjects whose future-session
outcomes were already used in earlier analysis. This is a forearm gesture
method check, not facial silent-speech evidence. The source is the
[Hyser corpus](https://physionet.org/content/hd-semg/2.0.0/).

The previous [grouped pilot](33-hyser-grouped-meta-pilot.md) represented
each one-second, 256-channel trial with only whole-trial log RMS and waveform
length. We added four ordered 512-sample bins, each with the same two
features. The result has ten features per recorded channel. The first two
exactly reproduce the old whole-trial features for every extracted trial;
the feature-cache builder checks this against the publisher-verified baseline
cache. Raw and derived arrays remain outside Git.

A new batched ridge scorer accepts any number of features per site. A
two-feature parity test matches the previous single-design reference,
including a day-2 class missing from day-1 training. Both feature families
use the **same fixed eight-site layout**. For each subject and feature
family, alpha is chosen by three day-1 stratified folds from 1, 10, 100,
and 1000; the fitted model is then scored on day 2. This separates feature
quality from electrode selection. The [complete result](results/hyser-temporal-fixed-dev13-v1.json)
records per-person CV curves, choices, outcomes, and source hashes.

| Subject | Whole-trial day 2 | Temporal day 2 | Difference |
| --- | ---: | ---: | ---: |
| 01 | 0.601 | 0.551 | −0.051 |
| 02 | 0.218 | 0.187 | −0.030 |
| 03 | 0.470 | 0.455 | −0.015 |
| 04 | 0.618 | 0.510 | −0.108 |
| 05 | 0.574 | 0.559 | −0.015 |
| 09 | 0.627 | 0.667 | +0.039 |
| 10 | 0.335 | 0.402 | +0.067 |
| 11 | 0.162 | 0.260 | +0.098 |
| 12 | 0.300 | 0.305 | +0.005 |
| 13 | 0.495 | 0.588 | +0.093 |
| 14 | 0.353 | 0.460 | +0.107 |
| 15 | 0.525 | 0.500 | −0.025 |
| 16 | 0.299 | 0.240 | −0.059 |

The mean paired change is **+0.008** balanced-accuracy points, with six
people helped and seven harmed. Across people, the Spearman correlation
between best day-1 CV improvement and day-2 improvement is about **−0.17**.
Four-bin timing therefore does not establish a reliable next-day gain.
Choosing temporal features for individuals using day-1 CV alone would be
unsupported. Alpha tuning also changes the older fixed baseline: on subject
05, whole-trial accuracy rises from 0.515 at alpha 10 to 0.574 with day-1
CV-selected alpha 1.

The next question is whether a **bounded, labeled new-session calibration**
can correct drift. Such an evaluation must consume designated first
repetitions for calibration and score only later repetitions. Its cost in
prompts, time, and user effort must be reported; it cannot be compared to
zero-calibration results without stating that difference. The seven reserved
Hyser subjects remain unacquired.
