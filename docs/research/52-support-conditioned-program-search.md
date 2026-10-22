# Support-conditioned program search: development and method-check protocol

The earlier 12.7-billion-program grammar is fast to search, but selecting
by day-one cross-validation alone transferred poorly. The
[fixed mixed comparison](51-fixed-mixed-calibration-method-result.md) also
failed after a fixed new-session calibration. This experiment asks a
different question: can the **same labeled calibration examples used to fit
the decoder** help choose the physical program before the remaining
session examples are scored?

## Development replay

For each of subjects 01–05, sample 5,000 unique full programs from the
four-grid edge/mode grammar with seed 42. Each program uses eight array
contacts, a common reference, and eight AFE channels. The grammar includes
both shared-reference potentials and digital adjacent differences. The
candidate sets contain complete programs; no partial-program beam score is
used. Selection scores are:

- `day1_cv`: three-fold class-stratified day-one balanced accuracy;
- `day1_to_support`: a day-one ridge model scored on the first recorded
  day-two repetition of every gesture;
- `half_cv_half_support`: their arithmetic mean, with no fitted mixing
  weight.

Every selected program is refit with day one plus those same labeled
support examples, using the previously developed weight-4 rule, and scored
on the remaining day-two repetitions. Support and query indices are
disjoint. The `day1_to_support` selection has severe multiple-comparison
risk because only one support example exists per class.

| Program selection | Mean query balanced accuracy, five people |
| --- | ---: |
| Day-one CV | 0.730 |
| Day one to support | 0.747 |
| Half CV, half support | 0.763 |
| Fixed mixed | 0.758 |
| Established fixed eight-site | 0.704 |
| Same contacts as half-and-half, all shared reference | 0.665 |

The complete search and scoring took 5.4–5.9 seconds per person on the
local machine, excluding no explicit warmup or controlled contention.
This is a development observation, not a validated hardware advantage.
The [source artifact](results/hyser-support-conditioned-search-dev05-v1.json)
records per-person programs, source hashes, outcomes, and wall times. The
[script](../../software/python/scripts/hyser_support_conditioned_search.py)
reproduces it from the external Hyser feature caches.

## Frozen method check

Use subjects 06–08 and 17–20. The primary rule is fixed now:
5,000 seeded complete candidates; choose the largest equal average of
day-one CV and day-one-to-support accuracy; deterministic tie break by
program tuple. Compare query balanced accuracy after identical weight-4
support fitting with the established fixed eight-site program. A useful
method-level gain requires mean paired improvement at least +0.05, at
least five of seven people improving, and no person losing more than
0.05. Report all subject results, the fixed mixed comparator, both
single-score selectors, and the all-shared-reference signals at the
selected program's same contacts.

The [two-stage runner](../../software/python/scripts/hyser_support_search_method_check.py)
first records source- and code-bound choices. The selection artifact is
committed and pushed before the score stage computes query accuracy. The
runner loads the day-two feature matrix and labels to determine the
classwise split, but passes only the support examples to the selector.
This is a retrospective software separation rather than a blinded data
custodian. These subjects and their
day-two data were used in previous Hyser experiments, so this check
cannot establish independent participant generalization.

Hyser is forearm gesture data. The first repetition per class is not a
globally chronological online calibration sequence. Digital subtraction
of preprocessed signals does not model analog front-end noise, contact
impedance, or wearable fit. Any result applies to this method replay,
not facial silent speech or a manufactured device.
