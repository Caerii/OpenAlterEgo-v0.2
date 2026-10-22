# A fast compositional search fails the future-session decision

We expanded the measured Hyser hardware-program grammar. In each of four
forearm grids, choose one of 112 adjacent contact edges, then two of three
available signals on those same contacts: two shared-reference potentials
and one digitally computed local difference. Each complete program uses
**eight array contacts plus a common reference, eight AFE channels**.
There are `336^4 = 12,745,506,816` possible complete programs under this
grammar. These are digital replays of publisher-preprocessed signals, not
recordings from a reconfigurable analog device or facial silent speech.

The [batched search implementation](../../software/python/openalterego/research/compositional_search.py)
keeps the top 16 partial programs after each grid. It scores 336, 5,376,
5,376 and 5,376 candidates at the four stages: 16,464 scored programs
per person. A comparator samples 16,464 distinct **complete** programs
from the same grammar and ranks them with the same day-1 three-fold ridge
CV. The counts are matched, while computational cost is not identical:
the beam scores smaller models at early stages. Both choices are then
trained on day 1 and evaluated on day 2. The established fixed eight-site
shared-reference program and an illustrative fixed mixed program are
included. All five people here are development subjects whose day-2
outcomes have been examined in earlier work.

| Subject | Beam choice, day 2 | Random choice, day 2 | Fixed mixed | Fixed eight-site |
| --- | ---: | ---: | ---: | ---: |
| 01 | 0.424 | 0.313 | 0.510 | 0.571 |
| 02 | 0.167 | 0.192 | 0.141 | 0.242 |
| 03 | 0.086 | 0.273 | 0.421 | 0.441 |
| 04 | 0.471 | 0.569 | 0.510 | 0.598 |
| 05 | 0.029 | 0.387 | 0.377 | 0.515 |
| **Mean** | **0.235** | **0.347** | **0.392** | **0.473** |

The [result record](results/hyser-compositional-search-dev05-v1.json)
contains source hashes, exact programs, per-stage evaluation counts,
day-1 CV and wall times. Mean day-1 CV is **0.922 for beam versus 0.902
for random**, yet the beam loses 0.112 day-2 accuracy on average. Beam
selection took about 4.0 seconds per person versus 5.8 seconds for
random selection in these single Windows-host runs. It is computationally
more efficient under this comparison but makes a worse engineering
decision. Subject 03 and 05 are severe failures.

The failure has two plausible and unresolved components. A partial
program's CV score is not guaranteed to predict how it will interact
with later grids, so a narrow beam may discard the eventual best full
program. Even if the full day-1 CV optimum were found, the earlier
[ranking audits](45-differential-program-method-holdout-result.md) show
that day-to-day transfer is weak. We cannot attribute the loss to only
one component from this experiment. A larger beam or Rust implementation
would accelerate search but would not by itself validate the objective.

The practical next search space should include **representation and
new-session calibration policies** alongside contact programs. Existing
[fixed-layout calibration evidence](37-hyser-fewshot-heldout-result.md)
shows much larger held-out method gains than layout selection, although
its Hyser class-block ordering limits deployment interpretation. A
physically realistic facial-array evaluation and an ordered calibration
protocol remain essential before a wearable design claim.
