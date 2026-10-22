# Fresh-fit histogram benchmark

The [benchmark records](results/gaddy-fresh-fit-benchmark.json) compare
scikit-learn histogram boosting settings on existing training data only. Each
of sessions 5-10 and 5-11 served once as the internal development session;
the remaining five sessions trained the activity classifier and speech-energy
regressor. Each candidate was fitted twice per split with reversed candidate
order on the second repeat. The scored prediction was the same soft-gated slow
energy contour used by the full probe. No model prediction cache was used.

| Setting | Median activity + speech fit | 5-10 contour MAE | 5-11 contour MAE |
| --- | ---: | ---: | ---: |
| 255 bins, 120 rounds (reference) | 7.10 s | 0.27322 | 0.31215 |
| 128 bins, 120 rounds | 7.73 s | 0.27240 | 0.31061 |
| **64 bins, 120 rounds** | **3.20 s** | **0.27284** | **0.31141** |
| 255 bins, 80 rounds | 4.30 s | 0.27710 | 0.31338 |
| 128 bins, 80 rounds | 5.73 s | 0.27634 | 0.31160 |

The 64-bin candidate retained all 120 rounds and had slightly lower contour
MAE than the reference on both internal splits. The median fit time was lower,
but individual timings varied substantially on this shared Windows host.
Reducing rounds worsened contour MAE on both splits for the 255-bin model.

After selecting 64 bins using those internal splits, one full fresh-fit
comparison on the previously opened 5-4 session yielded soft-gated contour
MAE 0.27134 versus the locked reference 0.27243, and all-frame MAE 0.29252
versus 0.29326. This is an exploratory check on a session already used in
project development, **not a new independent generalization result**. The
[candidate scientific output](results/gaddy-contour-fast-bins64-5-4.json)
and full-run profiles are preserved.

The full fresh-run sequence alternated candidate/reference/candidate/reference:

| Run | Total wall time |
| --- | ---: |
| [64-bin first](results/gaddy-contour-fast-bins64-5-4-profile.json) | 31.61 s |
| [Reference first](results/gaddy-contour-reference-fresh-paired-profile.json) | 49.13 s |
| [64-bin second](results/gaddy-contour-fast-bins64-5-4-repeat-profile.json) | 19.71 s |
| [Reference second](results/gaddy-contour-reference-fresh-repeat-profile.json) | 18.93 s |

These full-run timings are too variable to establish a reliable end-to-end
speedup. The probe therefore keeps `reference` as its default and exposes the
64-bin candidate with `--fit-preset fast-bins64`. The result and stage profile
record the preset. The existing content-addressed model cache keys include
the changed `max_bins` parameter, so a new preset triggers fresh fitting.

Next, measure fresh fits under controlled CPU load with process CPU time and
wall time, and test the candidate on a newly locked public-data session before
promoting it as a general scientific result or a faster default.
