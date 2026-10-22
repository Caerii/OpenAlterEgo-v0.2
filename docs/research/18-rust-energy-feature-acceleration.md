# Rust energy-feature acceleration

The repository's existing Rust/PyO3 accelerator now includes
`extract_energy_features` and `decode_gaddy_flac`. The first computes the
eight log RMS and eight log waveform-length EMG features plus log10 audio
RMS, using explicit frame centers shared with the Python reference. The
second decodes the corpus's mono 16-kHz 16-bit FLAC directly to float64.
A 20-trial, 3,420-frame parity check found maximum absolute differences of
**zero** for decoded audio, features and targets. The existing spike-scatter
API was also exercised after updating PyO3 and numpy bindings for Python
3.14.

The [bounded benchmark](results/gaddy-rust-feature-benchmark.json) timed
five repetitions, with decoding and in-memory framing timed separately.
Median framing times across 20 trials were 1.461 seconds for the original
Python path, 0.396 seconds for Python with unused pitch estimation disabled,
and 0.0168 seconds for Rust energy-only framing. Rust was **24× faster than
the fair Python energy-only reference**. Median FLAC decode times were
6.589 seconds through 20 FFmpeg subprocesses and 0.0757 seconds in Rust,
an **87× decode-stage gain**. Timings varied across repetitions on this
Windows host; these are local medians, not portable performance guarantees.

As an additional integration check, loading and framing all 71 eligible
trials of session 5-11 took 46.7 seconds on Python energy-only versus 1.35
seconds with the optional Rust backend in one paired run. Both outputs were
identical. This comparison includes FLAC decode and SciPy EMG filtering,
but excludes dictionary learning and model fitting. It is not a
whole-experiment or inference speedup.

The [contour probe](../../software/python/scripts/gaddy_contour_activity_probe.py)
now accepts `--feature-backend rust` after the extension is built and
installed; Python remains the default. The locked 5-4 result used the
committed Python path. Equivalent frame centers are necessary for exact
parity because floating-point boundary truncation can move an EMG window by
one sample. A future full-pipeline profile should determine whether further
Rust work belongs in filtering, matrix factorization or experiment
scheduling. The current gradient-boosted and NMF fitting stages remain in
compiled scikit-learn routines and need a separate profile before porting.
