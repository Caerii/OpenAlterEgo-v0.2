# Compositional screening of a million physical programs

The [million-program audit](87-million-program-search-limit.md) showed that
exact Rust scoring can cover a large pool but optimizing its current
selection score harder does not reliably improve future-day accuracy. We
tested a cheaper **sparse compositional surrogate**: can the measured
selection scores of a few complete programs reveal which edge and signal
mode choices are promising across the much larger grammar?

Each feasible program contains one of 336 adjacent-edge/mode options from
each of four recorded forearm grids. The
[model](../../software/python/openalterego/research/compositional_surrogate.py)
fits ridge-regularized additive option effects to the exact selection
scores of the first 5,000 seed-ordered programs. It predicts scores for
the 990,000 programs beyond the uniform control's 10,000-program prefix,
chooses 5,000 finalists, and
scores those finalists exactly with Rust. The better of the 5,000 seed
programs and 5,000 finalists is selected. A uniform-search control uses
the first 10,000 programs' exact scores. Both methods therefore spend
**10,000 exact candidate evaluations per person**; only the screened
method adds a cheap million-program model scan. Its candidate finalists
exclude the control's extra 5,000, so all exact evaluations are unique.

The [runner](../../software/python/scripts/hyser_additive_surrogate_search.py)
keeps the same day-one CV, two separate labeled day-two support
repetitions, final supported decoder fit, and disjoint later-day query
as the earlier audit. Query labels are used only after both programs
are fixed. A diagnostic fit on the first 4,000 seed programs predicts
the next 1,000 exact selection scores; the final model then fits all
5,000 seed scores. The query outcome never trains either model.

| Candidate seed | Uniform 10k query mean | Additive-screened 10k query mean | Paired gain | Wins / losses |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 0.8114 | **0.8391** | +0.0276 | 8 / 4 |
| 43 | 0.8136 | 0.8151 | +0.0015 | 6 / 6 |
| 44 | 0.8018 | **0.8304** | +0.0285 | 9 / 3 |

Across the three seeded pools, the mean paired gain is **+0.0192**.
Resampling the 12 *people* after averaging their three seed-specific gains
gives a descriptive 10,000-draw 95% interval **[+0.0069, +0.0322]**.
The 36 person-seed comparisons are correlated and are not 36 independent
subjects. All 12 people and query outcomes were already exposed in prior
method development. The interval is an exploratory uncertainty summary,
not a prospective confirmation.

The additive surrogate's mean held-out seed-score Spearman correlation is
0.803, 0.810, and 0.802 for seeds 42–44. For seed 42 it recovers the exact
million-pool **selection-score** winner for 7 of 12 people, and the mean
selection-score regret versus scoring the entire million is 0.00165. Yet
seed 43's later-day gain is nearly zero. The model predicts the current
score surface much better than it predicts whether that score will survive
the next recording day.

For seed 42 on this Windows host, sampling the shared million-program pool
took 19.8 s once. Per person, the mean measured components were 0.251 s
for the 5,000 shared seed exact scores, 0.315 s for the uniform control's
5,000 additional exact scores, 0.007 s for both diagnostic and final
additive fits, 0.304 s to scan the remaining pool, and 0.274 s to score
the screened finalists exactly. Thus the screened selection path after
sampling is about 0.84 s per person, versus a previously measured median
34.7 s for exact scoring of all million candidates across the same five
folds. These are separate local runs, not a controlled portable speedup.
Sampling cost and data loading are outside that per-person comparison.

The three result records retain candidate-pool and source hashes, split
indices, selected programs and physical contacts, exact selection scores,
query outcomes, held-out model diagnostics, and component timings:
[seed 42](results/hyser-additive-surrogate-1m-12-v2.json),
[seed 43](results/hyser-additive-surrogate-1m-12-seed43-v2.json), and
[seed 44](results/hyser-additive-surrogate-1m-12-seed44-v2.json).
The seed-42 pool hash matches the exact
[million-program audit](results/hyser-candidate-scale-1m-rust-12-v2.json).
A separate timing-only replay selected identical programs and outcomes
for all 12 seed-42 people.

This is a promising **computational search prior** that trades a tiny
sparse model for far fewer exact program scores at equal candidate cost.
Its coefficients are effects in a recorded-signal score model, not
identified muscle, tissue, electrode, or causal hardware parameters.
Hyser measures forearm gestures; digital differences do not establish
analog differential pickup, facial silent speech, or reseating physics.
The scientific next step remains a frozen evaluation on facial repeated
sessions and a selector that forecasts later-session *decision regret*,
including uncertainty and abstention.
