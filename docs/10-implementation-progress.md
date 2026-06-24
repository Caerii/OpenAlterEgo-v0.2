# Implementation Progress: Research Validation Fixes

**Last updated:** June 2026

> **Note:** Phases 3–6 and critical literature fixes described in earlier drafts are **implemented**. See `14-systematic-roadmap.md` for current priorities.

---

## ✅ Completed Phases

### Phase 1: Preprocessing Enhancements ✅

- `FilterSpec` + `PreprocessingMode`: **standard** (1–50 Hz), **clinical** (0.5–8 Hz), **wide** (20–450 Hz)
- Harmonic notching (60 Hz + 2nd/3rd harmonics)
- Offline (`preprocess_basic`) and streaming (`OnlinePreprocessor`) paths
- Tests: `tests/test_filters.py`, `tests/test_emg_config.py`

### Phase 2: User Management Foundation ✅

- `UserProfile` (immutable), `UserManager` (file-based CRUD)
- Tests: `tests/test_users.py`, `tests/test_cli_user.py`

### Phase 3: Calibration Workflow ✅

- `users/calibration.py` — `calibrate_user()`, quality checks, JSON reports
- `--strict-motion` / `CalibrationConfig.strict_motion`
- Tests: `tests/test_calibration.py`

### Phase 4: Adaptive Thresholding ✅

- `StreamDecodeConfig`: adaptive EMA threshold, SNR gate, baseline SNR
- `PredictionStabilizer` with diagnostics in token meta
- Server flags: `--adaptive-threshold`, `--baseline-snr-db`, `--no-snr-gate`
- Tests: `tests/test_streaming.py`

### Phase 5: User-Aware Training ✅

- `train.py --user-id`, checkpoint `user_id`, profile update
- `--touch-calibration-date`, `--emg-mode`
- Tests: `tests/test_integration.py`

### Phase 6: User-Aware Serving ✅

- `server.py --user-id`, profile-driven model + preprocess + decode config
- Online quality monitor, `quality_warning`, unknown-user + `--model` fallback
- Tests: `tests/test_server_ws.py`, `tests/test_pipeline_sim.py`

### Critical Literature Fixes ✅

| Issue | Status |
|-------|--------|
| Wide bandpass (20–450 Hz) | ✅ `dsp/filters.py` |
| Motion / SNR monitoring | ✅ `dsp/quality.py` |
| Per-user personalization | ✅ `users/*`, train, serve |
| Literature-aligned simulation | ✅ `sim/literature.py`, biophysical engine |
| Data collection + quality | ✅ `users/collect.py`, `DataCollectionSession` |
| CLI (`user`, `calibrate`, `collect`) | ✅ `cli.py` |
| Hardware DSL + pipeline binding | ✅ `hardware/`, `--hw-spec` on collect/serve/sim-dataset, `hw run` |

---

## 🚧 Remaining Work (by priority)

See **`14-systematic-roadmap.md`** for the full phased plan. Summary:

### Phase A — Simulation realism (no hardware)
- [x] Realism ladder: `off` → `wearable` → `tang` → `field`
- [x] Biophysical engine + `tang` preset as default for `sim-dataset`
- [x] SNR auto-calibration (`--snr-target-db`, Tang 18.9 dB)
- [x] Montage-aware forward pickup geometry
- [x] Correlated motion bursts + electrode contact steps
- [x] Numba + Rust batched scatter backends (`pool_numba`, `accel/`)
- [x] Extended benchmark sweeps up to 4 kHz (`sim-benchmark --extended`)
- [x] Dual-regime SNR calibration (`--snr-motion-target-db` 12.7 dB)
- [x] Parallel dataset shards (`sim-dataset --shards N --workers W`)
- [ ] Motion-regime SNR validation on long corpus samples

### Phase A — Validation (real data) ✅
- [x] A/B standard vs wide on real session (`dataset ab-preprocess`, Gaddy + Gowda)
- [x] Latency benchmark on real checkpoints (Gaddy p95 ~44 ms; Gowda p95 ~1.5 s @ 31 ch / 5 kHz CPU)
- [x] Window sweep on real sessions (Gaddy 400 ms; Gowda 900 ms for latency target)
- [x] Gaddy Zenodo download + import pipeline — `docs/15-gaddy-validation-results.md`
- [x] Gowda small-vocab download + train — `docs/16-gowda-validation-results.md` (val 16.4%, 12 words)

### Phase B — Hardware
- [x] Hardware DSL (`.oae.json`) + `--hw-spec` pipeline binding
- [ ] V0 benchtop BLE bring-up
- [ ] Electrode placement guide

### Phase C — Robustness
- [x] Optional motion gating in preprocess (`--motion-gate`, attenuates during high motion_index)
- [x] Per-channel SNR warnings in serve (`--weak-channel-warn`, `channel_quality` status)
- [x] Re-calibration detector (SNR vs baseline, 3 dB threshold)
- [x] Window size accuracy/latency sweep (`window-sweep` CLI)
- [x] Richer CLI progress for calibrate/train (`fit_epochs`, `--quiet`)

### Phase D — Model upgrades (research)
- [x] SE-ResNet / channel attention (`--arch se_resnet`, default for train/calibrate)
- [ ] Knowledge distillation
- [ ] Edge tiny model (SpeechNet-scale)

### Phase E — Open vocabulary (long-term)
- [ ] Phoneme labels, Seq2Seq/CTC, LM post-processing

---

## Testing Status

### ✅ Tests in place
- DSP, quality, filters, emg_config
- Users, calibration, collect, CLI
- Streaming, infer abstention, integration
- Sim (heuristic + biophysical + phonology + realism)
- Server WebSocket, pipeline sim
- Hardware DSL (`test_hardware_dsl.py`, `test_hw_pipeline.py`)
- Packet format, ring buffer, data split

### ⏳ Gaps
- Real BLE hardware integration tests
- Full-pipeline WebSocket test with trained checkpoint
- Latency regression in CI
- Unity client untested

Run: `cd software/python && python -m pytest`

---

## Backward Compatibility

All user-facing features remain **opt-in** (`--user-id` optional). Default behavior matches pre-personalization stack.

---

## Next Step

**Phase A validation is complete** on two public corpora (Gaddy + Gowda). Highest-value next work:

1. **Phase B1** — V0 benchtop BLE bring-up (requires hardware)
2. **Phase D4** — channel importance visualization on Gowda 31-ch checkpoint (literature Fig. 17 alignment)
3. **Phase D2** — knowledge distillation from ensemble (Lai 2023)
4. **Optional** — full 499-trial Gowda import + phoneme-aligned windows for higher accuracy baseline
