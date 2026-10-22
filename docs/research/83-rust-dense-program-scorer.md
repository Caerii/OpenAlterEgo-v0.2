# Exact Rust scoring makes large program searches practical

The [100,000-program audit](82-large-pool-scaling-and-facial-data.md) spent
about 71–74 seconds per person scoring candidate programs. We added an
**explicit Rust accuracy backend** to the existing batched ridge evaluator.
It gathers only each program's selected feature columns from the prepared
Gram and class cross-product matrices, solves the small regularized system
with a Cholesky factorization, predicts the validation trials, and reports
the same class-balanced accuracy as the NumPy reference. Independent
programs run through Rayon. The original NumPy backend remains the default;
the Brier diagnostic remains NumPy-only.

Build the optional extension in an active Python virtual environment:

```text
maturin develop --release --manifest-path software/python/accel/Cargo.toml
```

Then pass `backend="rust"` to
[`score_programs`](../../software/python/openalterego/research/dense_programs.py)
or `--backend rust` to the
[large-pool runner](../../software/python/scripts/hyser_candidate_scale_audit.py).
An unavailable requested extension raises an error; it does not silently
fall back to a different timed implementation.

The [per-candidate parity runner](../../software/python/scripts/benchmark_hyser_dense_rust.py)
compared every score for **5,000 programs × five prepared folds × three
participants = 75,000 fold-program scores**. All 75,000 scores matched
NumPy exactly, including the selected program index for each participant.
Its [record](results/hyser-dense-rust-parity-5k-3-v1.json) contains input
record hashes, the compiled native-binary hash, exact-difference counts,
and within-process single-run timings. A separate
[100,000-program Rust replay](results/hyser-candidate-scale-100k-rust-3-v1.json)
selected exactly the same programs, selection scores, and later-day
outcomes as the prior [NumPy replay](results/hyser-candidate-scale-100k-3-v1.json)
at 5,000, 20,000, and 100,000 candidates for all three people.

| Participant | NumPy scoring, 100k | Rust scoring, 100k | Ratio |
| --- | ---: | ---: | ---: |
| 01 | 73.9 s | 3.8 s | 19.4× |
| 05 | 71.3 s | 6.0 s | 11.8× |
| 18 | 72.3 s | 7.4 s | 9.7× |

The total scoring-time ratio across these three local runs is **12.6×**.
The 5,000-program within-process parity runs had an aggregate **11.0×**
ratio. These are workload-specific timings on one Windows host, not a
controlled cross-machine benchmark; load and thread scheduling varied.
Sampling, feature extraction, and model preparation are outside the reported
scoring times. The native scorer does not change the data, selection rule,
or measured later-day result.

This is a substantial compute gain for broad search and simulator sweeps.
It does not solve the scientific bottleneck: the current selection score
still transports imperfectly from limited support data to future trials.
Further compute is useful for uncertainty studies and constrained
counterfactual programs, but a device recommendation still needs measured
facial-session and eventually physical-intervention validation.
