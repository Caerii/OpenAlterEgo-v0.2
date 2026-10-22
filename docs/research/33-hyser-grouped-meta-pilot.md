# Grouped program search on 13 paired-day arrays

**Status:** development result, not a successful device recommendation. The
publisher-hash-verified [Hyser corpus](https://physionet.org/content/hd-semg/2.0.0/)
now covers subjects 01–05 and 09–16 in the external cache. Subjects 06–08
and 17–20 remain unacquired for a later, frozen evaluation. The extra eight
subjects added 6,476 verified files and about 3.45 GB of external raw data.
The extracted arrays are also outside Git.

We evaluated 1,000 seeded feasible eight-contact designs plus the fixed
layout for each person. The [matrix metadata](results/hyser-meta-dev13-v1.json)
lists the 14 day-1 predictors, source-record hashes, subjects, and external
matrix SHA-256. The code can regenerate the matrix from freely downloadable
source data. Predictors include day-1 CV, one-cell translation scores,
neighbor darkness, feature amplitude/variability, and within-grid spacing.
The label is each candidate's next-day balanced accuracy relative to the
fixed layout. No person's day-2 outcomes enter that person's predictors.

Leave-one-person-out fits use all candidate designs from the other 12 people
and choose the excluded person's highest predicted design. Two simple model
families were tested: standardized ridge (alpha 1000) and a regularized
histogram gradient booster (80 iterations, at most eight leaves). The
[full grouped result](results/hyser-meta-loo-dev13-v1.json) records chosen
indices, predicted advantages, actual advantages, and per-person ranking
correlations.

| Model | Mean actual advantage over fixed | People helped | People harmed | Worst loss | Predicted gains above 5 points |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ridge | −0.051 | 5/13 | 8/13 | −0.373 | 0/13 |
| Histogram gradient booster | −0.059 | 3/13 | 10/13 | −0.436 | 0/13 |

The model choices were explored after seeing outcomes in this development
cohort. The grouped folds prevent each excluded person's day-2 labels from
entering its own fit, but they are **not an untouched test of the research
process**. A five-point abstention margin makes both models choose the
fixed design for every person; this avoids the observed losses but establishes
no improvement. A donor-layout average across 12 people also beat the fixed
layout for only 3 of 13 excluded people in the
[donor audit](results/hyser-donor-prior-dev-13-v1.json).

The central problem is limited observability of future contact/session
conditions. More program evaluations, faster Rust kernels, a deeper model,
or a larger surrogate trained on these same predictors would not by itself
establish next-day design advantage. The next candidate should use additional
information that is available *before* a recommendation: richer temporal EMG
features, contact-quality measurements, explicitly bounded new-session
calibration, or a validated device-physics prior. Future-session information
must have a defined acquisition time and cannot be quietly estimated from
the eventual evaluation set.

The [scoring benchmark](results/hyser-batch-benchmark-v1.json) measured 256
layout outcomes on one warm-cached day pair, three repetitions. Batched ridge
scoring took a median 0.076 seconds versus 0.613 seconds for per-layout
fits, an 8.0× improvement with maximum score difference below 3×10⁻¹⁶.
This is a single-host solver microbenchmark; it excludes download, feature
extraction, and matrix preparation. It supports using batched sufficient
statistics for the current search scale, while leaving open a Rust port when
profiling identifies a larger compute bottleneck.

For the actual silent-speech target, the University of Bremen's
[CSL-EMG_Array corpus](https://www.uni-bremen.de/en/csl/research/silent-speech-communication/csl-emg-array-corpus)
is more relevant: it reports 40 facial electrode contacts in a 4×8 cheek
array and an eight-contact under-chin strip, with chained differential
channels. It includes audible and silent speech across 12 sessions from
eight speakers under CC BY 4.0. Its official
download requires submitting an email to receive a link. The dataset's
session structure and array geometry need inspection before asserting it can
support the same paired-day design-ranking experiment. Hyser remains a
forearm method test, not facial-device validation.
