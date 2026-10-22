# Rust feature cache and full contour stage profile

The existing session 5-4 contour comparison was replayed twice with the
optional Rust feature backend: once with an empty feature cache and once with
the same cache populated. Training sessions remained 5-5, 5-6, 5-8, 5-9,
5-10 and 5-11. The previously locked Python result was used only for parity;
this replay did not select a new model or open another evaluation session.

The cache stores decoded, filtered, framed feature and target arrays with
utterance offsets in an uncompressed NPZ outside Git. Its key binds the subset
path and manifest, source file sizes and modification times, feature backend,
pitch setting, Python feature code, and the Rust binary when used. Cache writes
replace a temporary file atomically. Arrays are loaded without pickle and
checked for shape, boundaries and finite values. A changed source or feature
implementation creates a different key. This is a local iteration cache, not
an immutable scientific data artifact or a cryptographic verification of raw
recordings; the source manifest carries the subset's SHA-256 hashes.

## Measured replay

Wall times are seconds from one cold and one warm run on this Windows host.
The [cold profile](results/gaddy-contour-profile-rust-cold.json) and
[warm profile](results/gaddy-contour-profile-rust-warm.json) give every stage.

| Stage | Cold | Warm |
| --- | ---: | ---: |
| Train and development feature load | 10.224 | 0.533 |
| Dictionary fit and transform/design | 4.758 | 3.791 |
| Activity fit and prediction | 6.941 | 7.226 |
| Six regression fits and predictions | 22.852 | 33.997 |
| **Whole run** | **45.220** | **45.924** |

All shared scientific fields in both Rust replays matched the
[locked Python result](results/gaddy-contour-activity-5-4-locked.json)
exactly, including the soft-gated 200 ms contour change MAE of 0.2724285.
The warm cache removed 9.69 seconds of feature loading, but total runtime did
not improve in this pair because regression fits took 11.15 seconds longer.
One pair of wall-clock runs cannot establish a whole-experiment speedup.
MiniBatchNMF also reached its configured 100-iteration limit in both runs;
this is the same fitting warning as the original comparison.

To reuse this for a new, separately authorized session comparison, invoke
`software/python/scripts/gaddy_contour_activity_probe.py` with the existing
`--train`, `--dev` and `--output` arguments plus:

```text
--feature-backend rust
--feature-cache-dir F:\Datasets\OpenAlterEgo\Gaddy\feature-cache-v1
--profile-output docs\research\results\<run>-profile.json
```

The Rust extension must be importable as `openalterego_accel`; see the
[accelerator README](../../software/python/accel/README.md). Keep the cache
directory outside the repository and do not commit its raw-derived arrays.
Omit `--feature-cache-dir` to run uncached. The scientific result and timing
profile are separate JSON files.

The measured next performance target is the variable fit-and-predict cost of
the six scikit-learn regressors, especially the latent and sparse controls.
Before replacing their compiled implementation with Rust, profile their
internal work and compare a candidate on output parity and repeated
end-to-end timing. The cache already makes feature-only experiments cheap;
the full locked comparison should continue to run all controls.

An optional [content-addressed model and dictionary cache](20-gaddy-model-cache-performance.md)
now makes identical replays faster while preserving fresh-fit behavior when
inputs or parameters change.
