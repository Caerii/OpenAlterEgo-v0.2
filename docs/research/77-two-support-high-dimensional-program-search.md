# Retrospective two-support high-dimensional program search

The [Hyser high-density EMG corpus](https://physionet.org/content/hd-semg/2.0.0/)
offers 256 recorded forearm sites over two days. The existing mixed
monopolar/digitally differenced grammar contains approximately **12.7
billion** feasible eight-signal programs, with eight array contacts, a
common reference and eight AFE channels per program. The earlier
[single-support method check](53-support-conditioned-program-search-result.md)
found a +0.035 mean paired gain over a fixed eight-site layout on seven
exposed people, below its declared +0.05 threshold. The
[support-rotation audit](54-support-rotation-stability.md) showed that
selected programs changed sharply with the support repetition.

We tested whether two separate labeled day-two repetitions per gesture
make the program decision more stable and useful. The
[runner](../../software/python/scripts/hyser_two_support_robust_search.py)
samples the same 5,000 complete programs with seed 42. It scores day-one
three-fold CV and two **separate** day-one-to-support transfer scores.
Rules select by half CV plus either the first, the mean of both, or the
worst of both transfer scores; day-one CV and mean support-only are
additional controls. Every selected program and both fixed layouts are
then fit with **both** support examples per class at total support weight
four and scored on the identical remaining day-two query trials. Thus the
first-only selection control has the same final decoder data as the
two-support rules. Support is chosen by classwise repetition rank and is
not a globally chronological online calibration prefix.

All 12 participants and day-two outcomes have been used in earlier
research. This is **retrospective development**, not a new person-level
validation. [Five-person](results/hyser-two-support-robust-dev05-v1.json)
and [seven-person](results/hyser-two-support-robust-method07-v1.json)
records retain source hashes, support/query indices, exact programs,
selection scores, query outcomes, candidate budget curves, and local
timings.

| Rule, equal eight-signal cost | Earlier five people | Later seven people | All twelve |
| --- | ---: | ---: | ---: |
| Half CV + mean of two supports | 0.813 | **0.827** | **0.821** |
| Half CV + first support only | **0.822** | 0.802 | 0.810 |
| Half CV + worst support | 0.819 | 0.812 | 0.815 |
| Day-one CV only | 0.771 | 0.791 | 0.783 |
| Fixed eight-site shared-reference | 0.774 | 0.773 | 0.773 |
| Same contacts as mean-two, all shared-reference | 0.722 | 0.720 | 0.721 |

The mean-two rule beats the fixed eight-site layout for **8/12** people.
Its pooled mean paired gain is **+0.048 balanced-accuracy points**;
a 10,000-draw participant-bootstrap interval is **[+0.015, +0.084]**.
The seven-person group's mean gain is +0.055 with five wins, while the
earlier five-person group has +0.039 with three wins. The gain over
first-support selection is only **+0.011** pooled, with participant
bootstrap interval **[−0.007, +0.032]**. The two-support rule therefore
does not establish that averaging support scores is reliably superior
to the simpler first-support selector.

The selected mixed programs exceed their own all-shared-reference
**same-contact** controls for all 12 people, by +0.100 mean in this
digital replay. That isolates the importance of which recorded signal
derivation is used **within the replay**. It does not estimate the
contact impedance, correlated analog noise, or CMRR of a built
front end.

## Candidate budget and compute

The same precomputed mean-two selection scores were replayed through
ten seeded nested permutations of each 5,000-program pool. These ten
orders share all people, programs and query trials; they are search
order sensitivity, not independent replication. Mean later-day query
balanced accuracy across the 12 people rises with candidate budget:

| Candidate programs considered | Mean query accuracy |
| ---: | ---: |
| 100 | 0.787 |
| 300 | 0.786 |
| 1,000 | 0.795 |
| 3,000 | 0.810 |
| 5,000 | **0.821** |

The 5,000-minus-100 paired participant mean is **+0.0345**, with a
participant-bootstrap interval of **[+0.0144, +0.0553]**. Selection
scores also rise, but here some of the extra optimization survives the
later day. The local runs took about **2.2 seconds per person** on
average for three 5,000-program score passes, selected-query fits, and
the budget diagnostic. This is a workload-specific observation, not a
portable speedup claim; the budget curve reuses precomputed scores and
does not measure the runtime of a true truncated search.

This is the strongest current evidence that broad program search can
produce a **useful decision** in a high-dimensional measured-signal
space under explicit calibration cost. The gain is modest and
participant-dependent. The corpus measures forearm gestures; its
digital pair subtraction cannot validate the physics of a facial
silent-speech wearable. A separate facial multi-session corpus with
spatially resolved contacts, or a controlled physical intervention,
must test whether this selection mechanism transfers before hardware
recommendations are warranted.
