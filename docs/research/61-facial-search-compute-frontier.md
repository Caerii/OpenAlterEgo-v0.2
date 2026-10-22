# Facial temporal search: compute and decision frontier

The [beam-16 development](59-facial-temporal-and-fast-search-development.md)
and [exposed-subject replay](60-fast-facial-search-posthoc-check.md)
suggested that partial-program scoring can reduce search work sharply.
We also evaluated beams of width 1 and 4 on the same six previously
exposed method subjects. All methods used identical four-bin temporal
features, ridge alpha 10, and training-block CV. Query blocks 21–30
were already inspected in previous experiments; the accuracy figures
below are **post hoc diagnostics**.

| Search | Mean train CV | Mean later-block query | Mean wall time | Evaluations |
| --- | ---: | ---: | ---: | ---: |
| Exhaustive six-channel | 0.937 | 0.915 | 26.85 s | 38,760 full |
| Beam 16 | 0.935 | 0.914 | 0.57 s | about 1,134 partial/full |
| Beam 4 | 0.928 | 0.920 | 0.16 s | about 341 partial/full |
| Greedy beam 1 | 0.916 | 0.915 | 0.054 s | 105 partial/full |

Thus beam 4 used roughly **114× fewer program evaluations** than
exhaustive search, with a 0.009 lower mean *selection* CV score.
Its mean query score was 0.005 higher on this exposed sample, which
does not prove better generalization. Greedy search was even cheaper,
but its training-CV regret was larger. The
[narrow-beam artifact](results/faceemg11-stepwise-narrow-method06-posthoc-v1.json)
records every subject's choice, accuracy, stage counts, and timing.

For a more comparable local timing check, we loaded subject 07 and
prepared identical temporal CV folds once, warmed each method, and
timed three repetitions of each search in the same process. Median
times were 32.184 s exhaustive, 0.760 s beam 16, 0.194 s beam 4,
and 0.034 s greedy: **42×**, **166×**, and **947×** speedups on that
machine and workload. Exact, beam 4, and beam 16 all selected a
program with the same 0.9394 train-CV score for that subject;
greedy selected 0.9030. Runtime varied across repetitions, so these
ratios are illustrative local measurements rather than portable
throughput guarantees. The [timing artifact](results/faceemg11-search-timing-sub07-v1.json)
contains all repetitions.

The useful engineering conclusion is specific: a compositional search
can explore a large discrete measurement-program space quickly, and
beam width exposes a measurable quality–cost tradeoff. The biological
conclusion is still limited. FaceEMG-11 records facial/ocular commands
within a session, with no measured electrode placement or silent speech.
No result here validates simulation of facial device physics or an
overwhelming end-to-end wearable advantage.
