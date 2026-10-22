# A compositional prior for choosing recorded channels

The [session-stability study](24-silentwear-session-stability.md) found that
top-ranked virtual montages often fail to remain top-ranked a day later. This
pilot asks whether a small, transferable model of *channel contributions* can
regularize that selection. It is a statistical prior over observed channels,
not a causal model of moving an electrode or a validated physiological model.

## Leave-one-subject-out design

For each S01–S03 target, the [prior diagnostic](results/silentwear-cross-subject-prior-v1.json)
scores 6,006 subset/normalization structures. The personal score is the lower
of ridge-proxy accuracy for session 1→2 and 2→1. The donor score is the mean
session-3 ridge-proxy accuracy of the **other two** subjects, each model fitted
on that donor's sessions 1 and 2. A 16-coefficient additive model smooths the
donor surface: intercept, normalization, and 14 channel indicators, with ridge
penalty 100. The candidate program has exactly six indicators on. A half-and-
half rule averages standardized personal and additive-donor scores.

For each method, the chosen program is determined without the target person's
session-3 score. The [second-stage result](results/silentwear-cross-subject-prior-logistic-v1.json)
binds to the choice file's SHA-256 and fits the chosen six-channel logistic
classifier on the target's first two sessions, at fixed C=1. It then scores
session 3. This is a retrospective development comparison: these subjects and
sessions have been examined in earlier pilots, and the methods were formed
after seeing those experiments. Leave-one-subject-out arithmetic does not
restore a fresh holdout.

| Selection rule | S01 | S02 | S03 | Mean session-3 balanced accuracy |
| --- | ---: | ---: | ---: | ---: |
| Personal cross-session ridge score | 0.2659 | 0.2867 | 0.3170 | 0.2899 |
| Other subjects' raw future score | 0.2935 | 0.2366 | 0.3169 | 0.2823 |
| Additive donor channel prior | **0.3691** | 0.2604 | 0.3070 | **0.3122** |
| Personal + additive donor, equal weight | 0.3059 | 0.2835 | 0.3176 | 0.3023 |
| Fixed index-spread, robust normalization | 0.3558 | 0.2588 | 0.2783 | 0.2976 |

The additive donor prior has the highest mean, but **S01 supplies the gain**.
It loses to personal selection on S02 and S03. The raw donor program scores are
less reliable: broad donor-target score-rank correlations are only 0.401,
0.400, and 0.265. Smoothing a 6,006-program surface to channel main effects
appears promising as a variance-reduction idea, not yet as an established
selection advantage. The fixed column was also selected for display after
earlier development comparisons, so it cannot serve as a preregistered control.

For a stronger fixed-layout comparator, select one of the previously defined
10 subset/normalization combinations by mean session-3 logistic accuracy of
the two **donor** subjects. This leave-one-subject-out fixed-panel procedure
chooses `alternate:recording_robust` for S01 and
`spread:recording_robust` for S02/S03. Its target scores are 0.2803, 0.2588,
and 0.2783 (mean 0.2724), calculated from the
[fixed-panel result](results/silentwear-fixed-two-day-v1.json). That comparison
remains retrospective and the panel is small.

## What to carry forward

The emerging hypothesis is that broad, transferable channel effects may be
more stable than the apparent optimum among thousands of overlapping subsets.
This is compatible with a compositional prior, but does not show causality:
channel identities, anatomy, contact, and acquisition effects remain
confounded. Test the hypothesis across more freely available people and days,
with the additive penalty and competing selectors fixed before examining their
future sessions. Report per-person gains, uncertainty, and decision regret.
Only then consider a held-out promotion of this prior into the main search.
