# openalterego_accel

Rust/PyO3 extension for high-throughput spike scatter in biophysical EMG
synthesis and energy-feature extraction for Gaddy research probes. PyO3 0.26
builds with Python 3.14 while preserving the project's Python 3.10 floor.

## Build

Requires [Rust](https://rustup.rs/) and [maturin](https://www.maturin.rs/).

```bash
cd software/python
uv sync --extra accel   # optional numba fallback

cd accel
maturin develop --release
```

Verify:

```bash
uv run python -c "import openalterego_accel; print('ok')"
uv run openalterego sim-benchmark --fs 2000 --synth-mode rust
```

## Usage

Set `synth_mode=rust` in biophysical stream config, or use `synth_mode=fast` for auto-selection (rust > numba > python).

## API

- `scatter_unit_multichannel(x, spike_idx, amp_base, env, m, wi, n, c, li)`
- `scatter_unit_delayed(x, spike_idx, amp_base, env, m, wi, di, n, c, li)`

Both operate in-place on `x` shaped `(n, c)` float32.

- `scatter_pool_batched(x, mu_idx, spike_t0, spike_amp, tpl, lengths, w, dly, n, c, has_delays)` — one call per chunk (preferred).
- `extract_energy_features(emg, audio, emg_sample_rate, centers)` returns
  `(features, log10_rms)` from contiguous float64 arrays. `centers` must be
  the exact frame-center array used by the Python reference; for Gaddy this
  is `np.arange(0.15, duration - 0.15, 0.05)`. This path omits pitch because
  the contour experiments do not use it. The Python research probe selects
  it with `--feature-backend rust` after the extension is installed.
- `decode_gaddy_flac(path)` decodes the corpus's mono 16-kHz 16-bit FLAC
  directly to the same float64 samples as the FFmpeg reference. Other FLAC
  formats are rejected rather than resampled silently.

The [benchmark script](../scripts/benchmark_gaddy_rust_features.py) checks
feature parity against the Python reference on real trials before reporting
in-memory framing time. It excludes FLAC decoding, filtering, dictionary
learning and model fitting, so its speedup is not an end-to-end speedup.
