# Reuse per-worker scratch in the exact program scorer

The [Rust dense-program scorer](83-rust-dense-program-scorer.md) used to
allocate six small arrays for **each program on each fold**: selected
columns, Cholesky factor, class coefficients, solve workspace, and
per-class hit and trial counts. For a 100,000-program, five-fold search,
that entails millions of short-lived allocations. The scorer now uses
Rayon's `map_init` to allocate scratch once per worker and reuse it across
programs. It still computes each complete program's regularized ridge fit
and class-balanced validation accuracy independently.

We built the release extension in the project virtual environment and
compared it with NumPy on the same seed-42 pool for subjects 01, 05 and
18: **75,000 fold-program scores**, zero differences, including every
selected index. We then replayed subject 01's nested 5,000/20,000/100,000
pool. All four reported choices, selection scores, and later-day outcomes
matched the prior Rust run exactly.

| Subject 01, 100,000 programs | Previous Rust build | Scratch reuse build |
| --- | ---: | ---: |
| Day-one three-fold scoring | 2.507 s | 1.859 s |
| Two support-fold scores, combined | 1.462 s | 1.021 s |
| Total scoring | **3.969 s** | **2.880 s** |

This is a **1.38× local scoring improvement** in one before/after run,
not a portable benchmark or a gain in predictive accuracy. Sampling,
feature loading, and fold preparation are outside these times. A second
scratch-reuse run scored in 2.818 s, illustrating run-to-run variation.
The baseline was run at parent commit `9cf29e8`; the final build's native
binary SHA-256 is
`c49d1184ad815cdba9facc191bf451fae0afc29c893e8f415c4a259c56091806`.
The comparison used
[`hyser_candidate_scale_audit.py`](../../software/python/scripts/hyser_candidate_scale_audit.py)
with `--subjects 01 --budgets 5000 20000 100000 --backend rust`, and
[`benchmark_hyser_dense_rust.py`](../../software/python/scripts/benchmark_hyser_dense_rust.py)
with `--subjects 01 05 18 --candidates 5000`.

This improves the cost of exact scoring for realistic candidate grammars.
It does not make the forearm proxy predict facial silent-speech outcomes or
repair the selection-to-future-session transport problem. The
[nested donor gate](85-nested-donor-decision-gate.md) still abstains on all
12 exposed people; new facial sessions remain the needed scientific test.
