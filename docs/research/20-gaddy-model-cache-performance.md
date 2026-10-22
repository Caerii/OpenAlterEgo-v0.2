# Cached model replay performance

The 5-4 contour comparison was replayed with the same six training sessions,
Rust features, and fixed model settings. The new optional `--model-cache-dir`
stores NMF components and per-utterance codes, plus activity and six regression
prediction arrays. Each entry is addressed by the exact numeric inputs, model
class and parameters, and scikit-learn version. It is stored outside Git,
written atomically and checked for shape and finite values when read. The
separate feature cache remains optional. A changed training or development
array, model parameter or library version misses the corresponding entry.

| Replay | NMF cache | Model prediction cache | Total wall time |
| --- | --- | --- | ---: |
| [First run](results/gaddy-contour-profile-model-cache-cold.json) | Unavailable | 0 of 7 hits | 21.37 s |
| [Model replay](results/gaddy-contour-profile-model-cache-warm.json) | Unavailable | 7 of 7 hits | 6.99 s |
| [NMF prime](results/gaddy-contour-profile-all-cache-prime.json) | Miss | 7 of 7 hits | 5.37 s |
| [Fully cached](results/gaddy-contour-profile-all-cache-warm.json) | Hit | 7 of 7 hits | **1.29 s** |

All four runs matched every shared scientific field of the
[locked Python result](results/gaddy-contour-activity-5-4-locked.json)
exactly. The fully cached run is about **16.6 times faster than the first
run of this sequence**, but it performs no new model fitting. These are
single wall-clock observations on a busy Windows host, not portable training
speedups. A genuinely new model setting still incurs its fit; the cache helps
repeated scoring, reporting and partially changed experiments.

Thread-count probes were inconclusive as an optimization and showed large
runtime variation: one OpenMP/BLAS thread took 54.6 s; four threads took
218.3 s; two earlier default-thread runs took 45.2 and 45.9 s. The four-thread
run coincided with other active Python processes. Reported energy metrics were
unchanged, although NMF coefficients differed by approximately `8e-14` in
the one-thread run. No thread-count default was changed on this evidence.

Use the cache with the existing probe arguments:

```text
--feature-backend rust
--feature-cache-dir F:\Datasets\OpenAlterEgo\Gaddy\feature-cache-v1
--model-cache-dir F:\Datasets\OpenAlterEgo\Gaddy\model-cache-v1
--profile-output docs\research\results\<run>-profile.json
```

The profile reports each cache hit. Omit `--model-cache-dir` for a fresh fit.
Neither local cache is a scientific receipt, nor does a warm replay validate
new training performance. The next training-performance experiment should
benchmark candidate fitting implementations under controlled machine load,
with equal model settings, unchanged evaluation sessions, output parity and
fresh-fit timings.

The first [training-session fresh-fit benchmark](21-gaddy-fresh-fit-preset.md)
now compares a smaller histogram preset with the reference settings.
