# Simulation realism ladder

Pragmatic path to close the **sim → real** gap without hardware, using progressively richer synthetic EMG.

**Implementation:** `software/python/openalterego/sim/`

---

## Realism presets (ladder)

| Preset | Use when | Models |
|--------|----------|--------|
| `off` | Unit tests, fast CI | White + AR(1) + drift only |
| `wearable` | Wet lab / light dry | Pink LF, mains H2, motion bursts, contact steps, per-ch electrode gain/DC |
| `tang` | **Default for wearable targets** | Tang 2025 motion/SNR regime + shared rigid motion + contact events |
| `field` | Stress / worst case | Stronger motion, ADC soft-clip |

Escalate: `off` → `wearable` → `tang` → `field`.

---

## Engines

| Engine | Speed | Fidelity |
|--------|-------|----------|
| `heuristic` | Fast | Band-limited spatial noise tokens (legacy) |
| `biophysical` | Slower | MUAP motor pool, forward pickup, recruitment (**recommended**) |

Hardware DSL defaults and `sim-dataset` CLI now default to **`biophysical` + `tang`**.

---

## SNR auto-calibration

Tang et al. (2025) report **18.9 dB** static and **12.7 dB** under motion. Use `--snr-target-db` to binary-search `noise_scale` before dataset generation:

```bash
cd software/python
uv run openalterego sim-dataset --out ./session \
  --sim-engine biophysical --realism tang \
  --snr-target-db 18.9 --minutes 2
```

Hardware presets (`v1_wearable_ble`, `tang_2025_headphone`) set `snr_target_static_db: 18.9` in `.oae.json` / built-in presets.

`meta.json` records `snr_calibration` and `quality_metrics` for QA.

---

## Montage-aware geometry

`sim/montage_geometry.py` maps literature montage sites → 1D pickup positions. Biophysical forward model uses montage geometry when `montage_name` is set (automatic via `--hw-spec`).

---

## Recommended workflow (no hardware)

```bash
# Hardware-bound, SNR-calibrated session
uv run openalterego hw run v1_wearable_ble --out ./session --user-id alice --seconds 120

# Or explicit
uv run openalterego sim-dataset --hw-spec v1_wearable_ble --out ./session --minutes 2
uv run openalterego calibrate --user-id alice --data ./session --fs 500 --emg-mode wide
uv run openalterego train --user-id alice --data ./session --fs 500 --emg-mode wide
uv run openalterego serve --source sim --hw-spec v1_wearable_ble --user-id alice
```

---

## Related

- Hardware DSL: [`08-hardware-dsl.md`](08-hardware-dsl.md)
- Tang SNR targets: `openalterego/sim/snr_calibration.py`
- Tests: `tests/test_sim_realism_ladder.py`

See [**10-neurobiophysical-emg.md**](10-neurobiophysical-emg.md) for the full neurobiophysical architecture (fiber types, batch synthesis, roadmap).
