# TODO: Remaining Implementation Work

This document outlines the remaining work to complete the research validation fixes, integrating insights from recent literature on silent speech interfaces.

**Status (Mar 2026):** Core DSP, quality, calibration, user CLI, **`collect sim|ble`**, user-aware train/serve (incl. **`--touch-calibration-date`**), adaptive + SNR gating, online quality (incl. **`--quality-every-n-chunks`**), unknown-user + **`--model`** serve fallback, **`--latency-log-every-windows`**, checkpoint `user_id`, integration + **`tests/test_server_ws.py`** (demo token + ping/pong), and `docs/USER_GUIDE.md` are in place ✅. Remaining items below are **enhancements**, not blockers for the vertical slice.

**Reference Documents:**
- `07-research-validation.md` - Technical parameter validation
- `08-action-items.md` - Quick reference action items
- `09-implementation-design.md` - Detailed architecture design
- `10-implementation-progress.md` - Current progress status
- `11-priority-changes.md` - 🔴 Critical priority changes identified from literature
- `12-references.md` - Complete reference list with citations and importance
- `13-data-collection-priorities.md` - Data realism and collection priorities

**⚠️ CRITICAL FINDING:** Recent literature (2021-2024) uses **20-450/500 Hz bandpass** filters, while we use **1-50 Hz**. See `11-priority-changes.md` for details.

**✅ COMPLETED (high level):**
- Wide / standard / clinical preprocessing; harmonic notch; streaming-aligned calibration & train
- `dsp/quality`, `users/*` (profile, manager, calibration, `DataCollectionSession` library)
- CLI: `user`, `calibrate`, **`collect`**; `train --user-id`; `serve --user-id`, adaptive threshold, SNR gate, `--online-quality`, `quality_warning` status
- `tests/test_calibration.py`, `test_streaming.py`, `test_cli_user.py`, `test_integration.py`, **`test_server_ws.py`**, `test_collect.py` (65+ tests in `software/python`)
- Checkpoints may include `user_id`; `LoadedModel.user_id`; `emg_signal_band_hz_for_quality`

---

## 🔴 CRITICAL PRIORITY TASKS (Must Fix for Base Functionality)

Based on analysis of recent literature, these are the **highest priority** changes needed:

### 1. Wide bandpass mode — **DONE** (see `USER_GUIDE.md`)

**Remaining:** rigorous **A/B evaluation** on real data (same session, standard vs wide where fs allows); publish numbers in a short results note.

### 2. Motion Artifact Detection & Handling (CRITICAL - Real-World Robustness)

**Issue:** Motion artifacts reduce SNR by 33% (18.9 dB → 12.7 dB), but we don't detect/handle them.

**Action:**
- [x] Add motion artifact detection (low-frequency drift monitoring) ✅ **COMPLETE**
- [x] Add signal quality monitoring (real-time SNR) ✅ **COMPLETE**
- [ ] Integrate motion detection into preprocessing pipeline (optional enhancement)
- [x] Optional strict abort: **`calibrate --strict-motion`** / **`CalibrationConfig.strict_motion`** ✅
- [ ] Improve high-pass filtering documentation (< 20 Hz removal already works)

**Priority:** 🔴 **CRITICAL** - Essential for real-world deployment

### 3. Per-user personalization — **core path DONE** (calibrate, train, serve, profiles, collect)

**Remaining:** formal **latency benchmarks** (p50/p95 script or CI budget); BLE **labeling UX** after `collect ble`; optional defaults when neither user nor model (intentionally strict today).

---

## Phase 3: Calibration Workflow

### 3.1 Core Calibration Module

**File:** `openalterego/users/calibration.py`

**Requirements:**
- Compute per-user confidence threshold from calibration data
- Calculate baseline SNR metrics
- Validate minimum sample requirements
- Generate calibration report

**Key Insights from Literature:**
- **Geometry paper (Gowda et al., 2024)**: Domain shift across individuals is a "change of basis" - eigenbasis vectors differ per person. This validates the need for per-user calibration.
- **Tattoo paper (Wang et al., 2021)**: LDA works well with small samples (10 repetitions per word). Our calibration should target 50-200 samples per token (as per `08-action-items.md`).
- **Headphone paper (Tang et al., 2024)**: Motion artifacts significantly degrade SNR (18.9 dB static → 12.7 dB motion). Calibration should include signal quality checks.

**Implementation Tasks:**
- [x] Create `CalibrationConfig` dataclass ✅ **COMPLETE**
- [x] Implement `calibrate_user()` function: ✅ **COMPLETE**
  - [x] Load calibration data (signals.npy + events.csv format)
  - [x] Run preprocessing (user's preferred mode: standard/clinical/wide)
  - [x] Train temporary model on calibration data
  - [x] Compute confidence threshold from validation predictions:
    - Use mean - 2*std of correct predictions
    - Ensure threshold is in [0.5, 0.95] range
  - [x] Calculate baseline SNR (using quality module)
  - [x] Validate minimum samples per token (warn if < 50)
  - [x] Generate calibration report (JSON + human-readable)
- [x] Add signal quality checks: ✅ **COMPLETE**
  - [x] Detect motion artifacts (low-frequency drift) - using quality module
  - [ ] Check electrode impedance (if available) - future enhancement
  - [ ] Validate channel consistency - future enhancement
- [x] Tests: `tests/test_calibration.py` ✅

**Files to Create/Modify:**
- `openalterego/users/calibration.py` (new)
- `tests/test_calibration.py` (new)

---

## Phase 4: Adaptive Thresholding — **DONE**

- `StreamDecodeConfig`: `adaptive_threshold`, `threshold_alpha`, clips, `baseline_snr_db`, SNR deficit fields
- `PredictionStabilizer`: EMA on confidence, SNR gate delta, `reset()`, token `meta` diagnostics
- Server: `--adaptive-threshold`, `--baseline-snr-db`, `--no-snr-gate`, `--online-quality`, `quality_warning` status
- Tests: `tests/test_streaming.py`

**Optional later:** pass `UserProfile` object into `StreamDecodeConfig` (today threshold comes from merged decode config / CLI).

---

## Phase 5: User-Aware Training — **DONE**

- `train.py`: `--user-id`, `--users-dir`, `--emg-mode`, checkpoint `user_id`, profile update
- `tests/test_integration.py` (checkpoint / calibration checks)

**Remaining:** policy for whether **`train --touch-calibration-date`** should also refresh **`calibration_samples`** / **`baseline_snr`** (today: date only; full metrics still from **`calibrate`**).

---

## Phase 6: User-Aware Serving — **DONE**

- `server.py`: `--user-id`, profile-driven model path, decode + EMG preprocessor, optional online quality + `quality_warning`, unknown user + `--model` continuation, `--quality-every-n-chunks`, `--latency-log-every-windows`
- **Remaining:** optional **full-pipeline** WebSocket test (sim + real `model.pt`); **documented latency budget** (p50/p95) vs literature targets.

---

## CLI Commands — **DONE** (`user`, `calibrate`, `collect`)

### 7.1 User Management Commands (reference)

**File:** `openalterego/cli.py`

**Implemented:**
- [x] `openalterego user create|list|show|delete` (`--users-dir`, `--json`, delete confirm / `-y`)
- [x] `openalterego calibrate` (maps to `CalibrationConfig` / `calibrate_user`; `--strict-motion`, etc.)
- [x] `openalterego collect sim|ble` → session folder (`signals.npy`, `events.csv` or empty columns, `session.json`)
- [x] `openalterego train` delegates to `ml/train.py` with `--user-id`, `--emg-mode`, `--touch-calibration-date`, etc.
- [x] `openalterego serve` delegates to `api/server.py` with `--user-id`, adaptive + SNR + online quality + throttle + latency log flags

**Files:**
- `openalterego/cli.py`, `openalterego/ml/train.py`, `openalterego/api/server.py`

---

## 🔴 CRITICAL: Bandpass Filter Range Issue

**See `11-priority-changes.md` for detailed analysis.**

**Summary:**
- **Our current:** 1-50 Hz (for silent speech envelope, based on AlterEgo 2018)
- **Recent literature (2021-2024):** 20-450/500 Hz (for actual EMG signals)
- **Impact:** We may be filtering out critical signal content above 50 Hz

**Required Action (HIGHEST PRIORITY):**
- [x] Add "wide" preprocessing mode: 20-450 Hz bandpass ✅ **COMPLETE**
- [x] Keep "standard" mode (1-50 Hz) for AlterEgo compatibility ✅ **COMPLETE**
- [x] Allow users to choose based on signal characteristics ✅ **COMPLETE**
- [x] Test and compare both modes on same data ✅ **COMPLETE**
- [x] Document when to use each mode — see `docs/USER_GUIDE.md` preprocessing table ✅

**Files Modified:**
- ✅ `openalterego/dsp/filters.py` - Added "wide" mode
- ✅ `openalterego/dsp/online.py` - Supports wider bandpass (already worked)
- ✅ `openalterego/users/profile.py` - Added "wide" to PreprocessingMode
- ✅ `tests/test_filters.py` - Added 6 tests for wide mode
- ✅ `tests/test_users.py` - Added test for wide mode in profiles

---

## Research Insights Integration

### Critical Findings from Literature Review

#### 1. Sampling Rate Considerations

**Current:** 250 Hz (validated by AlterEgo papers)

**Literature Range:**
- Geometry paper: **5000 Hz** (very high, for detailed analysis)
- Tattoo paper: **500 Hz** (sufficient for 92.64% accuracy)
- Headphone paper: **1000 Hz** (96% accuracy)
- Knowledge Distillation: **1000 Hz** (85.9% accuracy)

**Action Items:**
- [ ] Document that 250 Hz is intentionally lower (for silent speech envelope)
- [ ] Consider making sampling rate configurable for future hardware
- [ ] Note: Higher rates may improve accuracy but increase power/compute

#### 2. Bandpass Filter Refinement ⚠️ CRITICAL

**Current:** 1-50 Hz (standard), 0.5-8 Hz (clinical)

**Literature Range:**
- Tattoo paper: **20-500 Hz** (much wider!)
- Headphone paper: **20-450 Hz** (4th order Butterworth)
- Knowledge Distillation: **20-400 Hz** (10th order Butterworth)

**Key Insight:** Our 1-50 Hz bandpass is **much narrower** than literature. Recent papers consistently use 20-450/500 Hz and achieve 85-96% accuracy. This suggests we may be missing important signal content.

**Action Items (HIGHEST PRIORITY):**
- [ ] **Add "wide" mode: 20-450 Hz bandpass** (see `11-priority-changes.md`)
- [ ] Document why we use narrower band (silent speech vs. audible speech EMG)
- [ ] Test "wide" mode vs "standard" mode on same data
- [ ] Validate that our narrow band doesn't miss important information
- [ ] Make bandpass configurable per user

#### 3. Window Size Optimization

**Current:** 600 ms default

**Literature Range:**
- Geometry paper: **1.5s** for words, **100-400ms** sliding windows
- Tattoo paper: **2000ms** total (800ms before + 1200ms after trigger)
- Headphone paper: **3000ms** (3-second windows)
- Knowledge Distillation: **1500ms** (1.5s windows)

**Action Items:**
- [ ] Make window size configurable (already done, but document better)
- [ ] Consider longer windows (1-2s) for better accuracy
- [ ] Test trade-off: accuracy vs. latency
- [ ] Add user-specific window size to UserProfile

#### 4. Channel Count Validation

**Current:** 7-8 channels (from AlterEgo papers)

**Literature Range:**
- Geometry paper: **22 channels** (very high)
- Tattoo paper: **4 channels** (92.64% accuracy on 110 words!)
- Headphone paper: **4 channels** (96% accuracy on 10 words)
- Knowledge Distillation: **3 channels** (85.9% accuracy on 26 words)

**Key Insight:** 4 channels can achieve excellent accuracy! Our 7-8 channels may be overkill, but provides redundancy.

**Action Items:**
- [ ] Document that 4 channels is sufficient (per literature)
- [ ] Consider channel selection/importance analysis
- [ ] Add channel importance visualization (future work)

#### 5. Model Architecture Insights

**Current:** 1D CNN (OpenAlterEgoCNN)

**Literature:**
- Geometry paper: SPD matrix learning on Riemannian manifolds (advanced, but shows geometric structure exists)
- Headphone paper: **1D SE-ResNet** with adaptive channel weighting (96% accuracy)
- Knowledge Distillation: **ResNet1D** (85.9% accuracy, similar to ours)

**Action Items:**
- [ ] Our 1D CNN architecture is validated by literature ✅
- [ ] Consider adding SE (Squeeze-and-Excitation) blocks for adaptive channel weighting (future enhancement)
- [ ] Document that simple architectures work well (don't need complex models)

#### 6. Personalization Strategy

**Key Finding from Geometry Paper:**
> "Domain shift in sEMG signals due to combined effect of anatomical, physiological, and neural drive properties is characterized by a change of basis."

This **strongly validates** our per-user personalization approach. The paper shows:
- Eigenbasis vectors differ per individual
- Models trained with proper geometric priors can generalize, but per-user training improves accuracy
- Small models (9,734 parameters) can achieve good results with little data

**Action Items:**
- [ ] Emphasize per-user calibration in documentation
- [ ] Consider future work: geometric priors (SPD matrices) for better cross-user generalization
- [ ] Document that small models are sufficient (don't need huge networks)

#### 7. Motion Artifact Handling

**Key Finding from Headphone Paper:**
- Static SNR: 18.9 dB
- Motion SNR: 12.7 dB (significant degradation!)
- 4th order Butterworth filter (20-450 Hz) effectively suppresses artifacts

**Action Items:**
- [ ] Add motion artifact detection to calibration
- [ ] Consider adaptive filtering based on motion state
- [ ] Document motion artifact impact on accuracy

#### 8. Small Sample Learning

**Key Finding from Multiple Papers:**
- Tattoo paper: 10 repetitions per word sufficient for LDA
- Knowledge Distillation: 150 samples per class (26 classes = 3900 total)
- Geometry paper: Small models work with little data

**Action Items:**
- [ ] Validate our minimum sample requirements (50-200 per token)
- [ ] Consider data augmentation strategies (already in headphone paper)
- [ ] Document that small datasets are feasible

---

## Testing Requirements

### Unit Tests
- [ ] `tests/test_calibration.py` - Calibration workflow
- [ ] `tests/test_streaming.py` - Adaptive thresholding (extend)
- [ ] `tests/test_integration.py` - End-to-end user workflow

### Integration Tests
- [ ] Full user workflow: create → calibrate → train → serve
- [ ] Cross-user generalization test
- [ ] Signal quality degradation test
- [ ] Motion artifact handling test

### Performance Tests
- [ ] Latency benchmarking
- [ ] Memory usage profiling
- [ ] Inference speed on edge devices

---

## Documentation Updates

### Required Documentation
- [ ] User calibration guide (`docs/user-calibration-guide.md`)
- [ ] Parameter tuning guide (`docs/parameter-tuning.md`)
- [ ] Electrode placement diagrams (reference geometry paper)
- [ ] Hardware requirements specification
- [ ] Troubleshooting guide (signal quality issues)
- [ ] Literature comparison table (our parameters vs. papers)

### Update Existing Docs
- [ ] `README.md` - Add user management section
- [ ] `docs/02-data-collection.md` - Add calibration protocol
- [ ] `docs/04-ml-baselines.md` - Add per-user training notes
- [ ] `docs/06-protocol-xr.md` - Add user-aware serving notes

---

## Future Enhancements (Post-MVP)

### Advanced Features
- [ ] **SPD Matrix Representation** (from geometry paper):
  - Represent EMG as symmetric positive definite matrices
  - Use Riemannian geometry for better cross-user generalization
  - This is advanced but could significantly improve accuracy

- [ ] **SE-ResNet Architecture** (from headphone paper):
  - Add Squeeze-and-Excitation blocks for adaptive channel weighting
  - Improve robustness to variable skin-electrode coupling

- [ ] **Knowledge Distillation** (from KD paper):
  - Train ensemble of models
  - Distill to lightweight student model
  - Improve accuracy while reducing compute

- [ ] **Channel Selection/Importance**:
  - Analyze which channels are most important per user
  - Allow users to disable noisy channels
  - Visualize channel importance (like geometry paper Figure 17)

- [ ] **Multimodal Integration**:
  - Combine with IMU (head motion compensation)
  - Eye tracking for context
  - Audio feedback (bone conduction)

### Research Questions to Explore
- [ ] Validate our 1-50 Hz bandpass vs. literature's 20-400 Hz
- [ ] Test if 4 channels is sufficient (vs. our 7-8)
- [ ] Compare window sizes: 600ms vs. 1500ms vs. 2000ms
- [ ] Evaluate cross-user generalization with geometric priors
- [ ] Test motion artifact robustness in real-world scenarios

---

## Other Missing Tasks (From Literature Analysis)

### Signal Quality & Robustness

- [x] **SNR Calculation & Monitoring:** ✅ **COMPLETE**
  - [x] Real-time SNR computation (signal band / noise band)
  - [x] Baseline SNR from calibration
  - [x] Alert when SNR degrades below threshold (in DataCollectionSession)
  - Reference: Headphone paper shows 18.9 dB static, 12.7 dB motion

- [ ] **Electrode Impedance Monitoring:**
  - Track skin-electrode contact quality
  - Detect when electrodes need adjustment
  - Reference: Tattoo paper shows impedance changes over time
  - **Note:** Requires hardware support (future enhancement)

- [ ] **Channel Quality Assessment:**
  - Per-channel SNR calculation
  - Identify and flag noisy channels
  - Allow disabling problematic channels
  - Reference: Headphone paper uses adaptive channel weighting
  - **Note:** Can extend quality module to support per-channel metrics

### Data Collection & Validation

- [x] **Calibration Quality Checks:** ✅ **COMPLETE**
  - [x] Minimum sample validation (warn if < 50 per token)
  - [x] Signal quality validation during collection (DataCollectionSession)
  - [x] Motion artifact detection during calibration
  - Reference: All papers show per-user/subject training is essential

- [x] **Session Metadata Collection:** ✅ **COMPLETE**
  - [x] Session metadata with SNR, motion index, quality metrics
  - [x] Electrode placement notes (optional field)
  - [x] Collection date and duration tracking
  - [x] Quality warnings during collection

- [ ] **Session-to-Session Variability Handling:**
  - Track calibration dates (✅ done in UserProfile)
  - Detect when re-calibration is needed (compare current SNR vs baseline)
  - Handle electrode placement variations
  - Reference: Tattoo paper tests long-term wear (10+ hours)
  - **Action:** Add re-calibration detection logic

### User Experience

- [x] **Calibration Progress Feedback:** ✅ **PARTIALLY COMPLETE**
  - [x] Real-time signal quality indicators (DataCollectionSession)
  - [x] Quality warnings during collection
  - [ ] Show progress during data collection (CLI/UI needed)
  - [ ] Visual feedback on electrode contact (requires UI)
  - Reference: Tattoo paper emphasizes user-friendly calibration

- [ ] **Error Handling & Recovery:**
  - Graceful degradation when channels fail
  - Clear error messages for signal quality issues
  - Suggestions for troubleshooting
  - Reference: Headphone paper tests scenarios from 4 channels to 1
  - **Action:** Add channel failure detection and handling

### Documentation & Guides

- [ ] **Electrode Placement Guide:**
  - Visual diagrams of electrode locations
  - Muscle anatomy references
  - Placement repeatability tips
  - Reference: Geometry paper Appendix D describes muscle-electrode mapping

- [ ] **Troubleshooting Guide:**
  - Common signal quality issues
  - How to improve electrode contact
  - When to re-calibrate
  - Reference: All papers emphasize real-world robustness

- [ ] **Hardware Compatibility Guide:**
  - Supported hardware platforms
  - Minimum specifications
  - Electrode type recommendations
  - Reference: Papers use various hardware (OpenBCI, custom, COTS)

### Performance Optimization

- [ ] **Latency Benchmarking:**
  - End-to-end latency measurement
  - Breakdown by component (acquisition, preprocessing, inference)
  - Target: <500ms for natural interaction
  - Reference: Headphone paper optimizes for real-time

- [ ] **Model Size Optimization:**
  - Knowledge distillation for smaller models
  - Quantization for edge deployment
  - Reference: Knowledge Distillation paper shows 20.8x speedup

- [ ] **Power Consumption Analysis:**
  - Profile power usage per component
  - Optimize for battery-powered devices
  - Reference: Tattoo paper emphasizes wearable, low-power design

---

## Implementation Priority

### ✅ COMPLETED (Critical Priorities)
1. ✅ **Bandpass Filter "Wide" Mode** (20-450 Hz) - Implemented and tested
2. ✅ **Motion Artifact Detection & Handling** - SNR monitoring and motion detection complete
3. ✅ **Calibration Workflow Core** - `calibrate_user()` function implemented
4. ✅ **Realistic Simulation Data** - Updated frequency bands, quality metrics
5. ✅ **Data Collection Utilities** - `DataCollectionSession` with quality monitoring

### Done (formerly “remaining critical”)
1. **Calibration tests** — `tests/test_calibration.py` (incl. strict motion)
2. **CLI** — `user`, `calibrate`, **`collect`**
3. **User-aware training** — `train --user-id`, checkpoint `user_id`, **`--touch-calibration-date`**
4. **User-aware serving** — profiles, decode/EMG, online quality, fallbacks, latency logs
5. **Adaptive thresholding** — `StreamDecodeConfig` + server flags
6. **Re-calibration hints** — `quality_warning` + `re_calibration_suggested` when SNR drops vs baseline
7. **Integration tests** — `tests/test_integration.py`, **`tests/test_server_ws.py`**, **`tests/test_pipeline_sim.py`** (sim + checkpoint + online quality / channel meta)

### High priority (next engineering cycles)
- **Latency benchmark** script or documented procedure (not only log lines)
- **WebSocket client test** against **`run_pipeline`** (receive token JSON with optional **`weak_channels`**)

### Medium priority (usability)
- **Documentation** — keep `USER_GUIDE` / `software/python/README` aligned with new flags
- **Error handling** — clearer messages for empty `events.csv`, BLE timeouts
- **Progress feedback** — richer CLI output for long `calibrate` / `train` runs

### Low priority (research / future)
- **Advanced architectures** — SE-ResNet, knowledge distillation, per-class thresholds
- **Geometric priors** — SPD / Riemannian ideas from literature
- **Multimodal** — IMU, eye tracking, etc.

---

## Success Criteria

- [ ] All existing tests pass
- [ ] New features have >80% test coverage
- [ ] Backward compatibility maintained
- [ ] User calibration workflow functional
- [ ] Per-user models achieve ≥85% accuracy (target from literature)
- [ ] End-to-end latency <500ms
- [ ] Documentation complete and clear
- [ ] CLI commands work seamlessly

---

## References

**See `12-references.md` for complete reference list with citations, links, and detailed explanations of why each paper is important.**

**Quick Reference:**
1. **Kapur et al. (2018)**: AlterEgo foundation paper - IUI 2018
2. **Kapur et al. (2020)**: Clinical validation - PMLR/ML4H
3. **Wang et al. (2021)**: Tattoo-like electronics - npj Flexible Electronics 5:20
4. **Tang et al. (2024)**: Headphone-integrated SSI - IEEE TIM-25-01227
5. **Gowda et al. (2024)**: Geometry of sEMG signals - arXiv:2411.02591v1
6. **Lai et al. (2023)**: Knowledge Distilled Ensemble - arXiv:2308.06533v1

---

## Notes

- All phases maintain backward compatibility
- User management is optional (works without `--user-id`)
- Literature shows 4 channels sufficient, but our 7-8 provides redundancy
- ✅ **Wide bandpass mode (20-450 Hz) now available** - Addresses literature discrepancy
- Per-user personalization is **critical** (validated by geometry paper's "change of basis" finding)
- ✅ **Motion artifact detection implemented** - SNR monitoring and motion index calculation
- ✅ **Calibration workflow** — `calibrate_user()` + tests + CLI

## Important Implementation Notes

### Recently completed (snapshot)

1. **Wide / standard / clinical** — `PreprocessingMode`, filters, online preprocessor, `USER_GUIDE` table
2. **Quality** — `openalterego/dsp/quality.py`, `OnlineQualityMonitor`, tests in `tests/test_quality.py`
3. **Calibration** — `calibrate_user`, reports, **`--strict-motion`**, `tests/test_calibration.py`
4. **Collection & CLI** — `openalterego collect sim|ble`, `tests/test_collect.py`
5. **Serve** — profile + model fallbacks, online quality throttle, latency logs, `tests/test_server_ws.py`
6. **Test count** — run `uv run pytest` under `software/python/` (65+ as of Mar 2026)

### Next steps (backlog)

1. **BLE collection UX** — helpers or docs for building **`events.csv`** after `collect ble`.
2. **Server** — optional **`websockets`** 14+ migration (deprecation warnings); full-pipeline WS test; **latency budget** doc vs `<500 ms` goal.
3. **ML** — ResNet1D / SE blocks, distillation, per-class thresholds (research track).
4. **Evaluation** — A/B standard vs **wide** on **real** data (same session); publish numbers.
5. **Motion** — optional **preprocessing** gating (not only `strict_motion` on calibrate + warnings).
6. **Docs** — trim duplicate “research action item” paragraphs elsewhere in this file when they repeat `USER_GUIDE.md`.

### 📊 Data Realism & Collection Status

**✅ Completed:**
- ✅ Simulation updated to use realistic frequency bands (20-450 Hz default, adaptive to fs_hz)
- ✅ Quality metrics added to dataset generation (SNR, motion index)
- ✅ `DataCollectionSession` class + **`openalterego collect`** for sim (and BLE duration capture)
- ✅ Session metadata collection (SNR, motion, electrode info)

**🔄 In Progress / optional:**
- Richer metadata fields in **`calibrate`** reports (electrode notes from session JSON)
- BLE **`events.csv`** authoring workflow (manual or tool-assisted)

**📝 Important Notes:**
- Simulation now uses 20-450 Hz by default (matches recent literature)
- Frequency band auto-adapts to sampling rate (clamps to Nyquist)
- Quality metrics are computed and stored in metadata
- `DataCollectionSession` provides real-time quality feedback during collection

---

## Quick Reference: Critical Findings

**Addressed in codebase (see `USER_GUIDE.md` + `software/python/README.md`):**
1. **Wide** bandpass mode (with fs constraints) alongside **standard** / **clinical**
2. **Motion / SNR** — `dsp/quality`, online monitor, calibrate warnings, **`--strict-motion`**
3. **Per-user path** — profiles, calibrate, train, serve, **`collect`**

**📚 See Also:**
- `11-priority-changes.md` - Detailed analysis of critical issues
- `12-references.md` - Complete reference list with citations and importance
- `07-research-validation.md` - Technical parameter validation
