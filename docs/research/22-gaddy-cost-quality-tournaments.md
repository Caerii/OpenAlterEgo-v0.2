# Fresh-fit cost versus contour quality

Two tournaments used only the six existing training sessions, holding out
5-10 and 5-11 in turn. Both loaded the same provenance-keyed Rust features;
neither used the model prediction cache. Session 5-4 was **not** used to choose
or evaluate these candidates. Fit and prediction wall times and process CPU
times are retained in the [contour-head records](results/gaddy-head-tournament.json)
and [activity-gate records](results/gaddy-activity-tournament.json). These are
single runs per fold on a shared Windows host, so timing is exploratory.

## Contour heads

The activity classifier was fitted once per fold and shared across the heads.
Each head fitted only speech-active training frames; all used the same soft
activity gate and inactive target mean for scoring. The reference in this
tournament is the 255-bin histogram booster.

| Head | Fit time on 5-10 / 5-11 folds | Contour MAE on 5-10 / 5-11 |
| --- | ---: | ---: |
| Ridge, alpha 1 | 0.11 / 0.11 s | 0.30863 / 0.34265 |
| Ridge, alpha 100 | 0.11 / 0.11 s | 0.30844 / 0.34254 |
| Additive splines + ridge | 1.93 / 1.24 s | 0.29439 / 0.32620 |
| Random Fourier features + ridge | 1.50 / 0.40 s | 0.29410 / 0.33186 |
| Histogram boost, 64 bins | 2.27 / 3.51 s | 0.27319 / 0.31167 |
| Histogram boost, 255 bins | 3.34 / 1.45 s | 0.27322 / 0.31215 |

The linear head was fast, but its contour error increased by roughly 0.03.
The nonlinear inexpensive heads also lost roughly 0.014–0.020 absolute MAE.
None is a justified replacement for the boosted energy model. The 64-bin
booster retained quality, but fit wall time was not consistently lower across
these two folds.

## Activity gates

This tournament instead fixed the speech-energy model to the 64-bin booster
and changed only the activity classifier. The activity metrics and the final
soft-gated contour were both scored on each held-out training session.

| Gate | Fit time on 5-10 / 5-11 folds | Average precision | Contour MAE |
| --- | ---: | ---: | ---: |
| Logistic regression | 0.71 / 1.03 s | 0.95448 / 0.94762 | 0.31065 / 0.32935 |
| SGD logistic regression | 0.74 / 1.06 s | 0.95195 / 0.94505 | 0.31564 / 0.32620 |
| Histogram boost, 64 bins | 1.45 / 2.16 s | 0.97028 / 0.96940 | 0.27284 / 0.31141 |
| Histogram boost, 255 bins | 2.44 / 2.11 s | 0.97018 / 0.96973 | 0.27319 / 0.31167 |

The cheap gates lost both activity precision and contour quality. They are
not suitable substitutes for the current soft gate on this evidence.

## Decision

No cheap head or gate advances from this tournament. A future candidate
should first improve fit cost by at least 5× on repeated controlled runs
**and** stay within 0.005 absolute contour MAE of the boosted reference on
every internal session holdout. That threshold is a prospective promotion
rule established after these exploratory runs, not a preregistration of them.
The next scientific evaluation must use a separately locked public-data
session after selection.

The negative result changes the high-performance path: optimize the cost of
the useful nonlinear model and the number of full fits per scientific
decision. A cheap model can still be tested as a *ranking surrogate* for
candidate electrode designs, but these tournaments establish only prediction
quality on fixed recordings. They provide no evidence that it ranks unseen
physical interventions correctly. A 100× or 1000× gain in useful research
decisions remains a hypothesis until a cost-aware search loop beats a fixed
budget baseline on held-out decisions.
