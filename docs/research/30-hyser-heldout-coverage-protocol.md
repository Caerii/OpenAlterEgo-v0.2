# Hyser coverage search: frozen next-day method check

**Status:** protocol fixed after development on subjects 01–03 and before
acquiring any data for subjects 04–05. Hyser records forearm gestures, not
silent facial speech. The source is the [Hyser dataset](https://physionet.org/content/hd-semg/2.0.0/).

## Question and decision rule

Does an eight-channel search constrained to exactly two sites on each of four
recorded grids beat a fixed, spatially spread eight-channel layout on day 2?
The **primary endpoint** is the mean of subject-level day-2 balanced gesture
accuracy across subjects 04 and 05. Both strategies use the same whole-trial
log RMS and waveform-length features and the same day-1 ridge classifier
(alpha 10). We will report the paired difference in percentage points and
each subject separately. With two people this is a method gate, not an
estimate of population benefit. A positive mean with both subjects positive
would justify a larger, separately frozen evaluation; any other outcome does
not establish an advantage.

Secondary comparisons are unconstrained greedy, budget-matched random search
under both site constraints, and all 256 channels as an observed-information
reference. They are descriptive and cannot rescue a failed primary comparison.

## Selection and separation

For each subject, selection uses only session 1. Each candidate receives the
mean of three stratified trial folds. Greedy adds one site at a time with
within-grid minimum separation of two grid cells. The coverage variant limits
selection to two sites per grid. Each random comparator evaluates as many
complete feasible layouts as its greedy counterpart evaluates prefixes. The
fixed layout is determined by grid coordinates without labels. Feature
extraction, folds, alpha, width, seed 42, geometry rule, and comparator code
are fixed in the Git commit containing this protocol and
`hyser_compare_strategies.py`.

We will acquire session 1 for subjects 04 and 05, create an exclusive
selection JSON containing source hashes and exact channels, then commit and
push it before downloading their session 2 records. The scoring command
requires the recorded code hash and training source hash to match. Day 2 is
used once for this frozen comparison; subsequent changes using these outcomes
are development work and require new subjects for another gate. A Git commit
provides a public ordering record, but this is not independent
preregistration or a blinded external replication.

## Boundaries

This compares subsets of **already recorded** forearm electrodes. It does
not validate interpolation to unrecorded positions, facial anatomy, silent
speech decoding, device physics, or a causal effect of moving an electrode.
Session drift and missing nominal gesture classes may affect accuracy. We
will report missing classes and the next-day outcomes even if the proposed
constraint fails. No participant study or new physical experiment is involved.
