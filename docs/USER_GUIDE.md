# OpenAlterEgo Python — User Guide (concise)

## Install and environment

From `software/python/`:

```bash
uv sync --group dev
uv run openalterego --help
```

## Preprocessing modes (EMG bandpass)

| Mode       | Band (typical) | When to use |
|------------|----------------|-------------|
| **standard** | ~1–50 Hz   | AlterEgo-style silent-speech envelope; works at **250 Hz** and up. |
| **clinical** | ~0.5–8 Hz  | Narrowband / clinical-style signals; low sample rates OK. |
| **wide**     | ~20–450 Hz | Recent sEMG literature; requires **high sample rate** (Nyquist above the high cutoff — enforced in code, typically **fs ≥ ~920 Hz**). |

Training, calibration, and serving use the same **EMG mode** when you pass **`--user-id`** (from the profile) or set flags consistently.

## Typical workflows

1. **Synthetic data** (batch): `openalterego sim-dataset --out ./sess --minutes 2`  
   Or **stream-style capture**: `openalterego collect sim --out ./sess --user-id <id> --seconds 120`  
   The simulator follows **literature-aligned spectral presets** (see `openalterego/sim/literature.py`): default **`semg_literature_clamped`** uses a Tang/Wang-style **~20 Hz** lower cutoff with the **upper cutoff clamped to Nyquist** (at 250 Hz you cannot synthesize a true 450 Hz EMG band). Use **`--emg-paradigm alterego_envelope`** for Kapur-style **~1–50 Hz** token content, or **`--fs 1000 --emg-paradigm semg_literature_full`** for a **20–450 Hz** passband. Optional **`--no-ar1`**, **`--line-noise-uV`**, and **`--mains-freq-hz`** tune correlated LF noise and mains hum.  
   **Biophysical MVP:** `sim-dataset` and **`serve --source sim`** accept **`--sim-engine biophysical`**: **MUAP-shaped** impulses + **Poisson** firing with **token-modulated rates** and **label-dependent spatial routing** (`openalterego/sim/biophysical/`). This is a first step toward motor-unit–style models, not volume conduction or recruitment physiology yet.
2. **Register user**: `openalterego user create --user-id <id>`
3. **Hardware capture** (no labels): `openalterego collect ble --out ./raw --user-id <id> --seconds 60 --device-name ... --data-uuid ...` then label offline:

   ```bash
   # markers.csv: time_s,label  (or sample,label / start_s,end_s,label)
   openalterego collect label-events --session ./raw --markers ./markers.csv
   ```

   Then calibrate/train as usual. Check SNR drift vs profile: `openalterego user check-quality --user-id <id> --data ./raw`.
4. **Calibrate** (streaming-aligned model + threshold + baseline SNR):  
   `openalterego calibrate --user-id <id> --data ./sess --fs 250`  
   Optional: **`--strict-motion`** to abort when motion index is too high.
5. **Or train** (session folder):  
   `openalterego train --data ./sess --fs 250 --user-id <id> --preprocess-mode streaming --arch se_resnet`  
   Default architecture is **SE-ResNet** (Tang 2025-style channel attention). Use **`--arch cnn`** for the legacy baseline.  
   Optional: **`--touch-calibration-date`** to stamp profile **`calibration_date`** when refreshing the model only.

### Public EMG datasets (real data)

List known sources and import commands:

```bash
openalterego dataset catalog --out ./datasets/catalog.json
```

| Dataset | URL | Import |
|---------|-----|--------|
| **Gaddy silent speech** | [Zenodo 4064409](https://doi.org/10.5281/zenodo.4064409) (~3.9 GB tar) | `openalterego dataset import-gaddy --out ./sessions/gaddy --max-samples 50` |
| **Gowda small vocab** | [OSF YM5JD](https://doi.org/10.17605/OSF.IO/YM5JD) · [Paper](https://arxiv.org/html/2502.05762v2) | See [`docs/gowda/`](gowda/00-README.md) · `openalterego dataset import-gowda --download --out ./sessions/gowda_top30` |

After import, train with matching **`--fs`** (Gaddy: **1000**, Gowda: **5000**) and **`--emg-mode wide`** for OpenAlterEgo literature bandpass (paper uses **80–1000 Hz** — see [`docs/gowda/methods/02-preprocessing-protocol.md`](gowda/methods/02-preprocessing-protocol.md)):

```bash
openalterego train --data ./sessions/gaddy --fs 1000 --preprocess-mode streaming --emg-mode wide --arch se_resnet
```

**Gowda small-vocab** (67 words, 31 ch @ 5 kHz — recommended: full corrected import):

```bash
openalterego dataset import-gowda --download --out ./sessions/gowda_top30 --top-labels 30 --min-samples-per-label 8
openalterego train --data ./sessions/gowda_top30 --fs 5000 --emg-mode wide --arch se_resnet \
  --preprocess-mode streaming --segment-ms 900 --split-by auto
```

Documentation: [`docs/gowda/00-README.md`](gowda/00-README.md) · Legal/ethics: [`docs/gowda/legal/00-README.md`](gowda/legal/00-README.md)

6. **Serve**:  
   `openalterego serve --source sim --user-id <id>`  
   Optional: **`--online-quality`**, **`--quality-every-n-chunks`**, **`--channel-quality-meta`** / **`--weak-channel-warn`**, **`--motion-gate`**, **`--latency-log-every-windows`**, **`--adaptive-threshold`**, **`--baseline-snr-db`** (profile **`baseline_snr`** enables **`RecalibrationMonitor`**: sustained >3 dB SNR drop → **`quality_warning`** with **`re_calibration_suggested`**).

7. **Tune window vs latency**: `openalterego window-sweep --model ./sess/model.pt --session ./sess`  
   **Latency budget**: `openalterego latency-bench --model ./sess/model.pt`

## Troubleshooting

- **`Wide mode requires fs_hz >= 920`**: Use **standard** or **clinical**, or record at higher **fs**.
- **Unknown user**: Run **`user create`** first (or calibration will create a minimal profile).
- **No tokens / low confidence**: Train with **`--preprocess-mode streaming`** to match server preprocessing; align **`--segment-ms`** with **`--window-ms`**.

**Training throughput** (large sessions): preprocess + segment disk caches, AMP (default on CUDA), **`--num-workers`**, **`--mmap-signals`**, **`--compile`**. Benchmark: **`openalterego train-benchmark`**. See [`docs/18-training-scalability.md`](18-training-scalability.md).

See `software/python/README.md` for CLI flags. Engineering backlog: `docs/TODO.md` (kept in sync with implemented features).
