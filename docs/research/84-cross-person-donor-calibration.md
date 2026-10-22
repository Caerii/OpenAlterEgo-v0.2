# Better global ranking still misses the program decision

The [rank-transport audit](80-mixed-program-rank-transport.md) found weak
agreement between two-support selection and later-day performance. The
[Rust scorer](83-rust-dense-program-scorer.md) makes it practical to measure
all **5,000 candidate outcomes** for each of 12 people in a few seconds.
We tested whether outcomes from *other people* can calibrate the target
person's program ranking.

The [runner](../../software/python/scripts/hyser_donor_calibrated_program_search.py)
uses one identical seed-42 candidate pool and the same day-one/two-support
split as before. For each person and program, it computes day-one
cross-validation accuracy, mean transfer accuracy to two separately labeled
day-two support repetitions, and accuracy on disjoint remaining day-two
queries after the same supported decoder fit. The target query labels are
used **only after selecting its program**. The
[cross-person model](../../software/python/openalterego/research/donor_calibration.py)
fits a three-feature ridge predictor of candidate query deviations: target
day-one CV, target mean support transfer, and the mean query outcome of
other people for that exact program. When fitting each donor's training
row, that donor's own outcome and the outer target's outcome are excluded
from its prior. A unit test replaces the outer target outcomes with NaNs
and verifies unchanged predictions.

| Selection at eight-signal cost | Mean target query balanced accuracy |
| --- | ---: |
| Existing target two-support rule | **0.821** |
| Other people's query mean only | 0.825 |
| Cross-fitted donor-calibrated predictor | 0.803 |
| Fixed eight-site layout | 0.773 |

The donor-only minus existing paired mean is **+0.0033**, with a
10,000-draw participant-bootstrap interval **[−0.0276, +0.0372]**:
six wins, five losses and one tie. The learned predictor is **−0.0186**
below the existing rule, interval **[−0.0399, 0.0000]**: one win, four
losses and seven ties. Its largest individual loss is about −0.106.
The original two-support and fixed-layout query outcomes reproduce the
[earlier result](77-two-support-high-dimensional-program-search.md)
exactly for all 12 people.

There is a revealing metric mismatch. Mean candidate-rank Spearman
correlation with the target query rises from **0.427** for the original
score to **0.512** for the learned predictor, yet the learned *maximum*
is a worse engineering choice. Improving average ranking is insufficient
when one extreme candidate out of 5,000 is deployed. The useful objective
is lower **decision regret** on future sessions, with uncertainty and an
option to retain the fixed or existing program.

The [record](results/hyser-donor-calibrated-12-v1.json) retains source
hashes, a candidate-pool hash, all support/query indices, left-out donor
sets, fitted coefficients, selected programs and physical contacts,
controls, and target outcomes. The 12-person Rust replay took about
5.2 seconds locally, including feature-cache loading and all six scores
per candidate (three day-one folds, two support transfers, one query).
This is a workload-specific observation, not a general speed benchmark.

All 12 people and query outcomes have been inspected in earlier work.
The leave-one-person-out structure prevents direct **target-label leakage
within this run**, but the method choice itself is retrospective. These are
forearm gestures and digitally derived signal modes. No inference here
establishes facial silent-speech recognition, true analog contact behavior,
or a physical electrode intervention. An uncertainty-aware decision rule
needs to be frozen and evaluated on new people or recording sessions before
it can guide wearable design.
