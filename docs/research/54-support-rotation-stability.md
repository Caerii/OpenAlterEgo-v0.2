# Calibration support rotation: selected-design stability

The [support-conditioned search method check](53-support-conditioned-program-search-result.md)
showed +0.035 mean paired accuracy against a fixed eight-site layout,
below its predeclared +0.05 threshold. This diagnostic asks whether the
decision survives changing *which* one recorded repetition per gesture
is labeled for calibration.

## Replay

For the same seven exposed Hyser subjects, choose within-class
repetition rank 0, 1, or 2 as support, leaving every other repetition
as query. All subjects have at least three repetitions per gesture.
Each rotation repeats the same seeded 5,000-complete-program search,
using the arithmetic mean of day-one CV and day-one-to-support accuracy.
The query decoder uses the same support with weight 4. The fixed layout
and the rank-zero selected program are refit and scored on each
rotation's identical query trials. Rank-zero selected programs are
asserted equal to the previously pushed selection artifact.

| Support rank | Mean selected minus fixed | People improved | Worst person |
| --- | ---: | ---: | ---: |
| 0 | +0.035 | 5/7 | -0.030 |
| 1 | +0.024 | 5/7 | -0.082 |
| 2 | +0.047 | 6/7 | -0.041 |

The overall mean across the 21 subject/rotation evaluations was +0.035.
Subject 17 averaged -0.043 across rotations and lost 0.082 at rank 1.
Subject 20 averaged +0.118. These repeated evaluations share subjects
and queries, so they are not 21 independent validation cases.

The exact selected eight-signal program persisted from rank zero in
only **3 of 14** later subject/rotation comparisons. Physical contact
overlap with the rank-zero program was 0–8 of eight contacts, with
median 1.5. Yet choosing a new program for each support rotation gained
only **+0.0034** mean balanced accuracy over reusing the rank-zero
program while refitting its decoder to that rotation's support.
The search is therefore unstable at the design-decision level, and its
adaptation to individual support examples gave negligible added query
value in this diagnostic. Multiple near-equivalent or noisy maxima are
plausible explanations; this result alone does not distinguish them.

The [source artifact](results/hyser-support-rotation-method07-v1.json)
records source hashes, selected programs, physical contacts, overlaps,
and scores. The [replay script](../../software/python/scripts/hyser_support_rotation_stress.py)
and its split test reproduce the analysis from external feature caches.
The observed total wall time for all seven people and three rotations
was 26.4 seconds on the local machine; it is not a controlled benchmark.

## Next decision

Search should reward **stable performance across plausible calibration
examples and contact perturbations**, including a penalty for worst-case
loss and uncertainty in the selected layout. That can be evaluated
without increasing the number of candidate programs. A small set of
robust candidates should then be checked on a truly independent
facial/session dataset before any physical-design claim.

This is an exposed-subject retrospective forearm dataset. Classwise
support rotation is not a globally chronological protocol. Digital
differences of publisher-preprocessed channels do not establish analog
wearable behavior.
