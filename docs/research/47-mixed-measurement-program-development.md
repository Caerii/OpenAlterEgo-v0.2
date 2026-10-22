# Mixed measurement programs at matched physical contacts

The all-differential [frozen method result](45-differential-program-method-holdout-result.md)
failed against a strong shared-reference baseline. We tested a more flexible
but still physically countable program on Hyser development subjects 01–05.
For each of four 8×8 forearm grids, the program uses one shared-reference
channel and one adjacent digital difference involving that contact. It
therefore consumes **eight AFE channels, eight array contacts and one shared
reference contact**. The [search code](../../software/python/scripts/hyser_mixed_program_pilot.py)
samples 5,000 seeded programs and selects by day-1 three-fold ridge CV.
Day-2 balanced accuracy is scored with the same feature and classifier
pipeline as the earlier channel experiments.

The [result record](results/hyser-mixed-program-dev05-v1.json) compares
the chosen mixed program with three distinct references: an illustrative
fixed mixed program, the established fixed eight-site shared-reference
program, and **all eight shared-reference signals from the chosen program's
exact same array contacts**. The last comparison holds placement, contact
count, AFE count and common reference constant while changing the digital
measurement representation.

| Subject | Selected mixed | Same eight contacts, all shared-reference | Established fixed eight-site |
| --- | ---: | ---: | ---: |
| 01 | 0.522 | 0.449 | 0.571 |
| 02 | 0.241 | 0.241 | 0.242 |
| 03 | 0.578 | 0.389 | 0.441 |
| 04 | 0.466 | 0.569 | 0.598 |
| 05 | 0.279 | 0.319 | 0.515 |
| **Mean** | **0.417** | **0.393** | **0.473** |

The mixed representation helps at the same contacts for two people, ties
one, and harms two. It reaches a higher mean than its exact-contact
all-shared-reference counterpart by 0.024, driven substantially by subject
03's +0.189. The established fixed eight-site program still wins on four
of five people and on the cohort mean. The fixed mixed program averages
0.392; the selected mixed program averages 0.417, another small and
inconsistent development gain.

This is a retrospective digital replay of publisher-preprocessed **forearm**
signals. It does not measure the analog noise, contact impedance, reference
electrode behavior or mechanical repeatability of a mixed-mode wearable.
Nor does it establish facial silent-speech performance. It does show that
contact selection and measurement mode should be represented separately in
the experimental harness: a local difference can add useful information
for one person yet erase it for another. An engineering recommendation
requires a selector that predicts those person-specific effects across
future sessions and beats the fixed recorded-site reference.
