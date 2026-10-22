# A cheap class-signal prior is a candidate, not a validated layout rule

The [channel diagnostic](90-channel-drift-transport.md) found that
two-day class-signal strength ranks next-day channel signal more
consistently than past drift ranks next-day drift. We tested the simplest
decision this suggests: choose six recorded channels with the strongest
early class-specific feature patterns, normalized by early within-class
variance. We evaluate both raw and recording-robust preprocessing, then
use the lower of the two directional day-1/day-2 ridge accuracies to pick
one preprocessing mode. Only **two complete six-channel programs** need
exact scoring. Both the choice and a fixed six-channel comparator receive
the same balanced C=1 logistic classifier fit on sessions 1+2 and are
scored on session 3.

| Subject | Signal-prior six channels | Day-3 signal prior | Fixed spread six channels |
| --- | --- | ---: | ---: |
| S01 | 3, 6, 7, 8, 9, 13 | 0.3063 | **0.3558** |
| S02 | 2, 3, 7, 8, 9, 13 | **0.2887** | 0.2589 |
| S03 | 3, 4, 5, 6, 7, 10 | **0.3348** | 0.2783 |
| Mean | — | **0.3100** | 0.2977 |

The [runner](../../software/python/scripts/silentwear_signal_prior.py)
records each choice before loading session 3 and retains source hashes,
early per-channel scores, both directional selection scores, and outcomes
in the [result](results/silentwear-signal-prior-dev03-v1.json) and adjacent
per-subject selection receipts. The fixed spread comparator is a previously
defined panel member, but its earlier display in comparisons followed
inspection of the same development data. These three day-3 sessions and
even this rule's first exploratory outcome were inspected before this
formal replay. The local records establish code ordering and provenance;
they are **not** independent preregistration or a new holdout.

The rule's mixed result is scientifically useful. Six-channel selection
based on early class information may be more informative than searching
the extreme of the current ridge score, but S01 loses substantially to
the fixed subset. The class-signal statistic ignores redundancy between
channels and articulatory timing within a command. Its correlation with
future per-channel signal does not establish a causal muscle or electrode
model. The channels are fixed positions in SilentWear's recorded montage;
there is no measured contact intervention or unobserved placement claim.

We should freeze this inexpensive prior alongside the fixed layout and
the additive screen for evaluation on **new repeated-session facial-array
data**. Further tuning on these same three people would make the apparent
gain less interpretable.
