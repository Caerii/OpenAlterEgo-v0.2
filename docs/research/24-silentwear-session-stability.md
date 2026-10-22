# Why the virtual-montage search has not yet won

These are **retrospective development diagnostics** on SilentWear S01–S03.
Session 3 was already exposed in the preceding pilot. S04 stays sealed. No
number below is an independent holdout estimate or a claim about unrecorded
electrode positions.

## Rank transport is the bottleneck

The [rank-stability diagnostic](results/silentwear-rank-stability-v2.json)
fits the same ridge proxy on session 1 and scores all 6,006 recorded-channel
subset/normalization structures on sessions 2 and 3. It shares each subset fit
across the two full sessions and five individual session-2 batches. Program
rankings have moderate global correlation, but the extreme choices selected by
search frequently change:

| Subject | Spearman score-rank correlation | Top-16 overlap | Top-100 overlap | Session-3 ridge-proxy regret of session-2 winner |
| --- | ---: | ---: | ---: | ---: |
| S01 | 0.631 | 0 | 2 | 0.1042 |
| S02 | 0.817 | 5 | 31 | 0.0224 |
| S03 | 0.611 | 0 | 4 | 0.0440 |

These 6,006 overlapping subsets are correlated designs, not independent
replicates. The ridge-proxy regret compares the session-2 proxy winner with the
best *session-3 ridge proxy* score; it is not logistic-classifier regret or a
physical-design oracle. The S01/S03 top-16 overlap of zero is a direct warning
about selecting a noisy maximum from thousands of correlated candidates.

Penalizing standard deviation across the five recordings **within session 2**
did little. With penalties 0, 0.5, and 1 times batch standard deviation, the
future proxy regret stayed 0.1042 for S01, moved 0.0224 to 0.0090 for S02,
and moved 0.0440 to 0.0438 for S03. Same-session variability is an inadequate
stand-in for day-to-day drift on this evidence.

## A two-session selection rule

The [bidirectional search](results/silentwear-program-search-bidirectional-v5.json)
fits on session 1 and scores session 2, then reverses the roles. It ranks a
program by the *lower* of its two balanced accuracies. Both ridge screening
and the full logistic finalists use that rule. After recording the choices,
the selected program is refitted on sessions 1 and 2 together and scored on
session 3. Ridge-guided and canonical random search each receive 48 full
directional fits for selection; the 6,006 ridge screens cost approximately
9.5–11.0 seconds per subject in these single Windows-host runs.

| Subject | Ridge choice, session 3 | Random choice, session 3 | Predefined `spread:recording_robust`, session 3 |
| --- | ---: | ---: | ---: |
| S01 | 0.2685 | 0.2440 | 0.3558 |
| S02 | 0.3040 | 0.2839 | 0.2588 |
| S03 | 0.3170 | 0.3152 | 0.2783 |
| Mean | 0.2965 | 0.2810 | 0.2976 |

The fixed result comes from a [panel of five predefined subsets under two
normalizations](results/silentwear-fixed-two-day-v1.json), trained on the same
two sessions at regularization C=1. The `spread:recording_robust` column was
chosen for this table **after** inspecting that panel, so its mean is a
descriptive challenge, not a prospective baseline victory. Conversely, the
search was developed after earlier session-3 inspections. Neither can claim a
fresh test here. Channel index spread must not be interpreted as physical
electrode spacing without publisher geometry.

To regenerate these artifacts from the repository root, use the dataset Python
environment and set `PYTHONPATH` to `software/python`. All three commands need
`--root F:\Datasets\OpenAlterEgo\SilentWear`,
`--cache-dir F:\Datasets\OpenAlterEgo\SilentWear\event-cache-v1`, and the
matching `--output` path linked above:

```text
software/python/scripts/silentwear_rank_stability.py
software/python/scripts/silentwear_program_search.py --proxy ridge --selection-mode bidirectional --seed 42
software/python/scripts/silentwear_fixed_baselines.py
```

The earlier [one-direction search](23-silentwear-program-search-harness.md)
averaged 0.2641 for ridge and 0.2469 for random on session 3. The new run
improves both, largely because final fitting uses two sessions instead of one;
it also doubles directional-fit work. Comparing these runs does **not** isolate
the effect of bidirectional selection. Within the new equal-budget comparison,
ridge's margin is 0.0155 balanced-accuracy points over random, and it loses
to the displayed fixed baseline on S01.

## Decision

Do not promote this selector as a hardware-design engine. The next selection
experiment should use days as the uncertainty unit and compare against a
predeclared fixed-layout baseline, random search at matched compute, and a
simple session-robust selector. Preserve the final subject/session as a single
evaluation gate. A useful model must improve **future-session decisions** and
calibrate its uncertainty, not merely fit thousands of same-day programs.

The physics model should predict observable changes to the signal distribution
under a specified electrode or contact intervention. Existing public channel
subsets cannot identify those counterfactuals. Until geometry and intervention
data exist, the scientifically defensible work is to measure stable information,
noise, redundancy, and transfer limits within the recorded montage.
