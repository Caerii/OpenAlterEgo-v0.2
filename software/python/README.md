# OpenAlterEgo (Python)

Install and run from this directory with [uv](https://docs.astral.sh/uv/). For the full project overview, see the repository root `README.md`.

## Quick start

```bash
cd software/python
uv sync --group dev
uv run pytest
uv run openalterego --help
```

Synthetic EMG presets and citations: **`openalterego/sim/literature.py`**. `sim-dataset` supports **`--emg-paradigm`**, **`--no-ar1`**, **`--line-noise-uV`**, **`--sim-engine biophysical`** (MUAP + Poisson pool; see **`openalterego/sim/biophysical/`**).

## Users and calibration

```bash
uv run openalterego user create --user-id alice
uv run openalterego collect sim --out ./session_folder --user-id alice --seconds 120
uv run openalterego calibrate --user-id alice --data ./session_folder --fs 250
uv run openalterego train --data ./session_folder --fs 250 --user-id alice --preprocess-mode streaming
uv run openalterego serve --source sim --user-id alice
```

- **`collect`**: `collect sim …` records a session from the synthetic stream (writes `signals.npy`, `events.csv`, `session.json`). `collect ble …` records from BLE for a fixed duration (you usually add or align `events.csv` afterward).
- **`user`**: `create`, `list`, `show`, `delete` (optional `--users-dir`, default `./users`).
- **`train --user-id`**: saves `model.pt` under that user and updates `profile.json`.
- **`serve --user-id`**: loads `model_path`, confidence threshold, window/stride, EMG bandpass, and optional **`baseline_snr`** for decoding (override with `--model`, `--window-ms`, etc.).

### Adaptive threshold & SNR gate (serve)

- **`--adaptive-threshold`**: EMA-tracks recent softmax confidences and nudges the acceptance gate down when predictions are consistently high (above ~0.85), up when low (below ~0.60), and drifts back toward the calibrated **`--min-confidence`** in between. Clamped to **[0.5, 0.95]**.
- **`--threshold-alpha`**: EMA blend factor (default `0.1`).
- **SNR gate**: If **`baseline_snr_db`** is set (from **`--baseline-snr-db`** or the user profile’s **`baseline_snr`**), each decode step can add a gate boost when **`snr_db`** is present on the **`FrameChunk.meta`** dict (e.g. from your acquisition layer). Use **`--no-snr-gate`** to ignore profile SNR. Optional **`--snr-deficit-scale`** tweaks how many probability points to add per dB below baseline.

Emitted token JSON **`meta`** includes **`gate_threshold`**, **`eff_threshold`**, and **`snr_db`** when SNR was supplied on the chunk.

### Online quality in the server

- **`--online-quality`**: runs a sliding-window **`OnlineQualityMonitor`** on **preprocessed** samples, writes **`snr_db`** into chunk metadata (so the SNR gate and clients see it without a custom acquisition layer).
- **`--quality-warn-db`** / **`--quality-status-interval`**: when **`baseline_snr_db`** is configured, the server occasionally broadcasts a **`status`** message with **`quality_warning`** and **`re_calibration_suggested: true`** if SNR falls below baseline minus that margin (rate-limited).
- **`--quality-every-n-chunks`**: call the online quality monitor every N chunks and reuse the last SNR estimate between updates (lighter CPU load when **`--online-quality`** is on).
- **`--latency-log-every-windows`**: log mean per-window inference time every N classifier windows (0 = off).
- **`--channel-quality-meta`** (with **`--online-quality`**): add **`snr_db_per_channel`** and **`weak_channels`** (indices more than **`--weak-channel-deficit-db`** below median SNR) to chunk metadata, emitted **token** JSON **`meta`**, and **`quality_warning`** status when applicable.

If **`--user-id`** does not match a profile but **`--model`** is set, the server logs a warning and continues with decode defaults (no profile EMG / baseline SNR).

Training: **`--touch-calibration-date`** updates **`calibration_date`** on the saved user profile when **`--user-id`** is set (does not change **`calibration_samples`** / **`baseline_snr`**; use **`calibrate`** for those).

Calibration: **`--strict-motion`** / **`--motion-reject-above`** abort early when assessed motion is too high.

Checkpoints from **`calibrate`** or **`train --user-id`** include optional **`user_id`**; **`load_model`** exposes it on **`LoadedModel.user_id`**.

Broader usage and mode selection: **`../../docs/USER_GUIDE.md`** (repo root `docs/`).

## Hardware DSL (`.oae.json`)

Validate and simulate literature-aligned acquisition stacks before building PCBs:

```bash
uv run openalterego hw list
uv run openalterego hw validate v0_openbci
uv run openalterego hw resolve v1_wearable_ble --json
uv run openalterego hw simulate v0_openbci --path both --seconds 3

# End-to-end: validate → smoke sim → collect session
uv run openalterego hw run v0_openbci --out ./session --user-id alice --seconds 60
uv run openalterego collect sim --hw-spec v0_openbci --out ./session --user-id alice
uv run openalterego serve --source sim --hw-spec v0_openbci --user-id alice
```

Specs live in **`../../hardware/specs/`**; docs in **`../../hardware/08-hardware-dsl.md`**.
