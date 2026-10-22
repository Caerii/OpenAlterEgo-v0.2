# Does searching more measured pair programs help a later day?

The [first measured-pair pilot](42-measured-differential-pair-pilot.md) used
one seeded pool of 1,000 feasible eight-channel, 12-contact programs per
person. We then generated 5,000 feasible programs per person and measured a
nested search-budget curve. All work here is retrospective development on
Hyser subjects 01–05. It concerns digitally differenced, publisher-preprocessed
**forearm** recordings, not facial silent speech or analog device behavior.

The [audit code](../../software/python/scripts/hyser_differential_budget_audit.py)
computes day-1 three-fold ridge CV and a day-1-fit/day-2-test balanced
accuracy for each program. Fifty seeded permutations of the same 5,000
candidate pool make budgets 1, 10, 30, 100, 300, 1,000, 3,000 and 5,000
nested. Each budget selects the maximum day-1 CV within its prefix. The
permutations assess sensitivity to candidate order; they are **not** fifty
independent people or recording sessions. The [result record](results/hyser-differential-budget-dev05-v1.json)
contains individual curves, source hashes and the fixed-pair comparator.

| Programs considered | Mean selected day-1 CV | Mean selected day-2 accuracy |
| ---: | ---: | ---: |
| 1 | 0.769 | 0.200 |
| 10 | 0.840 | 0.244 |
| 30 | 0.862 | 0.247 |
| 100 | 0.883 | 0.244 |
| 300 | 0.896 | 0.247 |
| 1,000 | 0.906 | 0.271 |
| 3,000 | 0.916 | 0.301 |
| 5,000 | 0.918 | 0.309 |

The fixed differential program averages 0.249 day-2 accuracy. Searching more
programs improves this five-person mean, unlike the failed
[information proxy](40-information-proxy-real-data-falsification.md). But the
result is heterogeneous: at the full 5,000-program budget, selected day-2
accuracies are 0.265, 0.167, 0.388, 0.502 and 0.225 for subjects 01–05.
The first and third remain below their fixed differential baselines (0.505
and 0.403). Subject 04's weak center-fixed baseline and large search gain
substantially influence the cohort mean. The earlier strong fixed eight-site
monopolar benchmark still exceeds the selected differential score for all
five people.

The scientific decision is to test **one frozen 5,000-program rule**, with
the same fixed differential and strong recorded-site references, on a
separate method-evaluation cohort. These people have been used for other
Hyser studies, so the cohort is not an untouched person-level validation.
The differential pair outcomes and search-budget method have not yet been
evaluated there. A positive average alone will not establish a hardware
advantage: we need paired per-person gains, absence of large regressions,
and comparisons at the real contact/AFE costs. A later facial dataset and
eventual physical intervention remain necessary for wearable claims.
