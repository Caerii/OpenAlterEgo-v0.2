# Realism ablation harness

Sweep **realism presets** (`off`, `wearable`, `tang`, `field`) and **Tang SNR calibration**
targets against real OSF `gowda_sv_full` before committing GPU time to full sim-transfer runs.

## CLI

```bash
cd software/python

# Fast probe ladder (signal stats vs real train events, ~minutes)
uv run openalterego analyze sim-realism \
  --real ./sessions/gowda_sv_full \
  --probe-only

# Probe + sim-only + anchor transfer on all variants (GPU hours)
uv run openalterego analyze sim-realism \
  --real ./sessions/gowda_sv_full \
  --transfer \
  --trials 100 \
  --device cuda

# Transfer only top-3 probe matches (recommended first GPU pass)
uv run openalterego analyze sim-realism \
  --real ./sessions/gowda_sv_full \
  --transfer \
  --top-k 3 \
  --trials 100 \
  --pretrain-epochs 30 \
  --anchor-epochs 15 \
  --device cuda
```

Report: `sessions/gowda_sv_full/ablations/realism_ablation/realism_ablation_report.json`

Per-variant sim corpora: `.../realism_ablation/corpus/<tag>/`

## Default variant grid

| Tag | Preset | SNR static / motion (dB) |
|-----|--------|--------------------------|
| `off_raw` | off | — |
| `wearable_cal` | wearable | 18.9 / 12.7 |
| `tang_cal` | tang | 18.9 / 12.7 (default Gowda sim) |
| `field_cal` | field | 18.9 / 12.7 |
| `tang_nocal` | tang | — |
| `tang_lo_snr` | tang | 15 / 10 |
| `tang_hi_snr` | tang | 22 / 15 |

Subset: `--variants tang_cal,wearable_cal,off_raw`

## Probe metrics

Per variant, generate a short Gowda-shaped biophysical corpus (`--probe-trials`, default 8) and compare
event-segment statistics to real OSF **train** events:

- **SNR** — time-domain band power (80–1000 Hz real, literature token band sim)
- **Motion index** — low-frequency drift proxy
- **Channel RMS** — after gowda preprocessing (model-facing)
- **Correlation** — mean |off-diagonal| channel correlation after gowda preprocessing

Lower `match.total` = closer sim→real probe match. Variants are ranked by this score.

## Transfer metrics

For each selected variant: generate `--trials` corpus → train **sim-only** CTC (real SPD basis) →
**anchor finetune** on real → evaluate real test WER (trial LM decode).

See also [09-sim-transfer.md](./09-sim-transfer.md).

## Code

- `openalterego/sim/metrics/realism_match.py` — probe statistics
- `openalterego/ml/eval/sim_realism_ablation.py` — orchestration
