# openalterego_accel

Rust/pyO3 extension for high-throughput spike scatter in biophysical EMG synthesis.

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
