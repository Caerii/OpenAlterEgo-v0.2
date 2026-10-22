# Additive program screening does not improve next-day silent commands

The [million-program Hyser result](88-additive-million-program-screen.md)
showed that a sparse compositional model can screen many measured forearm
programs with few exact evaluations. We tested the same *search idea* on
[SilentWear](https://huggingface.co/datasets/PulpBio/SilentWear), which
contains repeated-day silent command recordings. Its 14 recorded channels
support 3,003 six-channel subsets. Two preprocessing choices make **6,006
virtual programs**. These are subsets of recorded signals, not measurements
at new electrode sites.

The [runner](../../software/python/scripts/silentwear_additive_screen.py)
uses sessions 1 and 2 for selection. It scores a program with the **lower**
of two directional class-balanced ridge accuracies: fit day 1/score day 2,
and fit day 2/score day 1. Per subject and candidate-order seed, 150 exact
program scores train a ridge-10 additive model of fourteen channel-inclusion
effects plus preprocessing mode. It screens the remaining 5,706 programs,
then scores 150 finalists exactly. The control scores the first 300
seed-ordered programs exactly. Both policies therefore have a **300-program
exact-score budget**. A diagnostic model trained on the first 120 scores
predicts the next 30, without using the final session.

Each choice is written to an exclusive local selection record before the
session-3 loader is called. Both selected programs then receive the same
balanced C=1 logistic classifier fit on sessions 1+2 and are evaluated on
session 3. The records are an ordering and lineage guard, not an
independent preregistration: all three subjects' later sessions have been
examined in earlier research.

| Candidate seed | Uniform 300 day-3 mean | Screened 300 day-3 mean | Paired mean gain |
| ---: | ---: | ---: | ---: |
| 42 | 0.2928 | 0.2899 | −0.0029 |
| 43 | 0.2817 | 0.2901 | +0.0084 |
| 44 | 0.2904 | 0.2843 | −0.0061 |

The mean gain over the nine subject-seed comparisons is **−0.0002**.
They represent only three people, not nine independent participants.
The held-out surrogate rank correlations with the *selection score*
average 0.722, 0.685 and 0.809 for seeds 42–44. The screened choice's
selection score is higher by about 0.0056, 0.0162 and 0.0154 respectively
on average. Better approximation and optimization of this two-day proxy
does **not** yield a dependable day-3 classifier gain here.

The three [result records](results/silentwear-additive-screen-dev03-seed42-v1.json)
([seed 43](results/silentwear-additive-screen-dev03-seed43-v1.json),
[seed 44](results/silentwear-additive-screen-dev03-seed44-v1.json)) and
their adjacent per-subject selection receipts retain source manifest
hashes, session/batch provenance, selected channels and normalization,
selection scores, future-session outcomes, and component timings. The
selection records contain no day-3 outcome. This is an offline event-level
command benchmark; complete unlabeled batches are used for recording-robust
normalization, and streaming interaction latency is not measured.

This negative transfer is informative. The additive search prior can
efficiently find high-scoring combinations within two observed sessions.
The bottleneck is a **validated predictor of later-session utility** under
calibration, contact, and placement variation. The public virtual-subset
data cannot identify which of those physical causes dominates, and the
result cannot establish a facial wearable's electrode intervention effect.
Future search methods should earn deployment by reducing paired
future-session decision regret, with uncertainty and a fixed-program
fallback, on a fresh repeated-session facial array corpus.
