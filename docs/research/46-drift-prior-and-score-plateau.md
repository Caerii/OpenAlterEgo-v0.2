# Donor drift prior and score-plateau audit

We tested a compositional prior for the measured Hyser differential pairs.
For each of 12 development people (01–08, 17–20), it compares class-mean
pair features on days 1 and 2, scales their shift by day-1 variation and
class signal, and ranks pair stability within each recorded grid. A target
person's prior averages the **other eleven** people's ranks. Candidate
programs are scored by day-1 CV minus a penalty times the mean donor rank
of their eight pairs. Target day-2 labels score the chosen program only.

The [leave-one-person-out record](results/hyser-pair-stability-loo-dev12-v1.json)
shows no convincing gain. At penalties 0, 0.01, 0.02, 0.05, 0.1 and 0.2,
mean day-2 balanced accuracy is 0.346, 0.347, 0.347, 0.347, 0.347 and
0.355. The largest penalty helps three people and harms three relative to
zero penalty; the other six are unchanged. The fixed differential program
averages 0.304 in this cohort. These are retrospective development results,
not fresh validation of a donor prior. The prior's basic assumption—that
the same array-relative pairs have transferable drift risk across arms—has
not earned trust.

We also audited ties at the top day-1 CV score. Among 5,000 candidate
programs, the [tie record](results/hyser-differential-ties-dev12-v1.json)
finds three tied maxima for subject 01, two for 03 and 04, and unique
maxima for the other nine. The three subject-01 ties reach day-2 accuracies
from **0.040 to 0.371**. Ties are resolved by canonical candidate order in
the frozen [seven-person method test](45-differential-program-method-holdout-result.md).
The earlier [search-budget development curve](43-differential-search-budget-development.md)
randomized candidate order within each repeated draw. Consequently, its
full-budget endpoint averages across *different tied winners* for subjects
01, 03 and 04, even though all 5,000 programs are present each time. That
curve measures both budget and tie-order sensitivity; it must not be
interpreted as the exact outcome of the canonical frozen selector. The
unique top scores for 06–08 and 17–20 mean tie order cannot explain the
large subject-06 loss in the frozen test.

The next program family should compare measurements at matched contact and
AFE cost, keeping the strong shared-reference fixed layout in view. A
plausible family uses one shared-reference channel and one local adjacent
difference on the same two contacts in each of four grids: eight channels,
eight array contacts plus the shared reference. This is a **digital replay**
of measured signals; analog contact and AFE physics remain untested.
