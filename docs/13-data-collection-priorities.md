# Data Collection & Realism: Priorities and Status

This document outlines what we've done to ensure realistic data and comprehensive data collection, plus remaining priorities.

---

## ✅ Completed: Data Realism Improvements

### 1. Simulation Frequency Bands Updated

**Issue:** Simulation used outdated (2-45 Hz) frequency bands, not matching recent literature.

**Solution:**
- Updated `TokenProfile` default `band_hz` from `(2.0, 45.0)` to `(20.0, 450.0)`
- Added `token_band_hz` parameter to `SimStreamConfig` (default: None = auto-select)
- Auto-adapts to sampling rate:
  - fs_hz >= 920 Hz: Uses full 20-450 Hz (wide mode)
  - fs_hz >= 100 Hz: Uses 20 to (Nyquist - 10 Hz)
  - Lower fs_hz: Uses 1 to (Nyquist - 5 Hz) for standard mode

**Files Modified:**
- `software/python/openalterego/sim/stream.py`
- `software/python/openalterego/sim/dataset.py`

**Impact:** Simulation data now matches frequency content from recent literature (2021-2024).

---

### 2. Quality Metrics in Dataset Generation

**Issue:** Generated datasets lacked quality metrics for validation.

**Solution:**
- Added quality metrics computation to `generate_dataset()`
- Computes SNR, motion index, baseline wander, signal/noise power
- Stores in `meta.json` for each generated dataset

**Metadata Now Includes:**
```json
{
  "quality_metrics": {
    "snr_db": 18.9,
    "motion_index": 0.15,
    "baseline_wander": 0.05,
    "signal_power": 0.85,
    "noise_power": 0.12
  }
}
```

**Files Modified:**
- `software/python/openalterego/sim/dataset.py`

---

### 3. Real-Time Data Collection with Quality Monitoring

**Issue:** No real-time quality feedback during data collection.

**Solution:**
- Created `DataCollectionSession` class in `openalterego/users/data_collection.py`
- Real-time quality monitoring using `OnlineQualityMonitor`
- Automatic quality warnings (low SNR, high motion)
- Comprehensive session metadata collection

**Features:**
- Real-time SNR and motion artifact detection
- Quality warnings during collection
- Session metadata (electrode placement, types, notes)
- Quality metrics stored in session.json

**Usage:**
```python
session = DataCollectionSession(
    user_id="alice",
    fs_hz=250,
    channels=8,
    preprocessing_mode="wide"
)

# Add chunks and get real-time quality feedback
for chunk in data_stream:
    metrics = session.add_chunk(chunk.samples)
    if metrics.snr_db < 10.0:
        print("Warning: Low SNR detected!")

# Add events
session.add_event(start_sample, end_sample, "yes")

# Finalize and save
session.save(output_dir)
```

**Files Created:**
- `software/python/openalterego/users/data_collection.py`

---

## ✅ Completed: Metadata Collection

### What We Now Collect

1. **During Calibration (`calibrate_user()`):**
   - Baseline SNR (dB)
   - Motion artifact index
   - Samples per token
   - Validation accuracy/loss
   - Confidence threshold
   - Calibration date
   - Warnings (insufficient samples, high motion)

2. **During Data Collection (`DataCollectionSession`):**
   - Real-time SNR (updated per chunk)
   - Real-time motion index
   - Session metadata:
     - User ID, session ID
     - Sampling rate, channels
     - Collection date, duration
     - Preprocessing mode
     - Electrode placement notes (optional)
     - Electrode types (optional)
     - Quality metrics (final)
     - Warnings/notes

3. **In Dataset Generation (`generate_dataset()`):**
   - Quality metrics (SNR, motion, power)
   - Simulation parameters
   - Token frequency bands used
   - Generation timestamp

---

## 🔴 Remaining Priorities

### 1. Calibration Tests (HIGHEST PRIORITY)

**Status:** Core function implemented, needs testing

**What's Needed:**
- [ ] Test `calibrate_user()` with synthetic data
- [ ] Test different preprocessing modes (standard/clinical/wide)
- [ ] Test threshold computation accuracy
- [ ] Test quality metrics collection
- [ ] Test report generation
- [ ] Test edge cases (insufficient samples, poor quality)

**Files to Create:**
- `tests/test_calibration.py`

**Why Critical:** Calibration is core functionality - must be tested before CLI integration.

---

### 2. CLI Integration (HIGH PRIORITY)

**Status:** Functions implemented, CLI commands missing

**What's Needed:**
- [ ] `openalterego user create --user-id <id>` - Create user profile
- [ ] `openalterego user list` - List all users
- [ ] `openalterego user show --user-id <id>` - Show user profile
- [ ] `openalterego user delete --user-id <id>` - Delete user
- [ ] `openalterego calibrate --user-id <id> --data <dir> [options]` - Run calibration
  - Options: `--preprocessing-mode`, `--min-samples`, `--tokens`
- [ ] `openalterego collect --user-id <id> [options]` - Start data collection session
  - Real-time quality feedback
  - Save to user directory

**Files to Modify:**
- `software/python/openalterego/cli.py`

**Why Critical:** Users need easy way to create profiles, collect data, and calibrate.

---

### 3. User-Aware Training (HIGH PRIORITY)

**Status:** Training script exists, not user-aware

**What's Needed:**
- [ ] Add `--user-id` parameter to `train.py`
- [ ] Load user profile and use user's preprocessing mode
- [ ] Save model to user directory
- [ ] Update user profile with model path
- [ ] Use user's window_ms for segment size

**Files to Modify:**
- `software/python/openalterego/ml/train.py`

**Why Critical:** Training must be per-user for personalization to work.

---

### 4. User-Aware Serving (HIGH PRIORITY)

**Status:** Server exists, not user-aware

**What's Needed:**
- [ ] Add `--user-id` parameter to `serve` command
- [ ] Load user profile and model
- [ ] Use user-specific threshold from profile
- [ ] Use user's preprocessing mode
- [ ] Use user's window_ms and stride_ms

**Files to Modify:**
- `software/python/openalterego/api/server.py`
- `software/python/openalterego/cli.py`

**Why Critical:** Serving must use per-user settings for accuracy.

---

### 5. Adaptive Thresholding (MEDIUM PRIORITY)

**Status:** Threshold computation done, adaptive adjustment missing

**What's Needed:**
- [ ] Load threshold from user profile in `PredictionStabilizer`
- [ ] Monitor recent confidence distribution
- [ ] Adjust threshold based on signal quality (SNR)
- [ ] EMA-based smooth threshold updates
- [ ] Clip to [0.5, 0.95] range

**Files to Modify:**
- `software/python/openalterego/runtime/streaming.py`

**Why Important:** Improves accuracy by adapting to signal quality changes.

---

### 6. Re-Calibration Detection (MEDIUM PRIORITY)

**Status:** Baseline SNR stored, comparison logic missing

**What's Needed:**
- [ ] Compare current SNR vs. baseline SNR from profile
- [ ] Detect significant degradation (>3 dB drop)
- [ ] Suggest re-calibration when needed
- [ ] Track time since last calibration
- [ ] Alert user when re-calibration recommended

**Files to Create/Modify:**
- `openalterego/users/calibration.py` - Add `check_recalibration_needed()`
- `openalterego/runtime/streaming.py` - Monitor and alert

**Why Important:** Signal quality degrades over time - users need to know when to re-calibrate.

---

### 7. Per-Channel Quality Assessment (MEDIUM PRIORITY)

**Status:** Overall quality done, per-channel missing

**What's Needed:**
- [ ] Compute per-channel SNR
- [ ] Identify noisy channels
- [ ] Flag channels with poor quality
- [ ] Allow disabling problematic channels
- [ ] Visualize channel importance

**Files to Modify:**
- `openalterego/dsp/quality.py` - Add per-channel functions
- `openalterego/users/calibration.py` - Track channel quality

**Why Important:** Some channels may be noisy - adaptive channel weighting improves robustness.

---

## 📊 Data Collection Checklist

### What We Collect ✅

- [x] Signals (signals.npy)
- [x] Events (events.csv)
- [x] Session metadata (session.json)
- [x] Quality metrics (SNR, motion index)
- [x] Calibration reports (JSON + text)
- [x] User profiles (with calibration metadata)
- [x] Baseline SNR
- [x] Motion artifact index
- [x] Samples per token
- [x] Validation accuracy/loss
- [x] Confidence threshold

### What We're Missing ⚠️

- [ ] Electrode impedance (requires hardware support)
- [ ] Per-channel quality metrics (can add)
- [ ] Session-to-session comparison (can add)
- [ ] Electrode placement photos/diagrams (manual)
- [ ] Environmental conditions (temperature, humidity)
- [ ] User notes/observations (can add to SessionMetadata)

---

## 🎯 Implementation Order (Next Steps)

### Week 1: Testing & CLI
1. **Write calibration tests** (1-2 days)
   - Test with synthetic data
   - Test all preprocessing modes
   - Test edge cases

2. **CLI integration** (2-3 days)
   - User management commands
   - Calibration command
   - Data collection command

### Week 2: User-Aware Pipeline
3. **User-aware training** (1 day)
   - Add --user-id to train.py
   - Use user's settings
   - Save to user directory

4. **User-aware serving** (1 day)
   - Add --user-id to serve
   - Load user profile and model
   - Use user's threshold

5. **Adaptive thresholding** (1-2 days)
   - Integrate into streaming
   - Signal quality-based adjustment

### Week 3: Enhancements
6. **Re-calibration detection** (1 day)
7. **Per-channel quality** (1-2 days)
8. **Integration tests** (1 day)

---

## 📝 Important Notes

### Data Realism
- ✅ Simulation now uses realistic frequency bands (20-450 Hz default)
- ✅ Quality metrics match literature values (SNR ~18.9 dB static)
- ✅ Motion artifacts simulated (drift, baseline wander)
- ✅ Frequency bands auto-adapt to sampling rate

### Data Collection
- ✅ Real-time quality monitoring available
- ✅ Comprehensive metadata collection
- ✅ Quality warnings during collection
- ✅ Session metadata stored with all data

### What's Next
1. **Test calibration** - Ensure it works correctly
2. **CLI integration** - Make it user-friendly
3. **User-aware pipeline** - Complete personalization
4. **Adaptive features** - Improve robustness

---

## References

- **Tang et al. (2024)**: SNR values (18.9 dB static, 12.7 dB motion)
- **Wang et al. (2021)**: Frequency bands (20-500 Hz)
- **Gowda et al. (2024)**: Per-user personalization importance
- See `12-references.md` for complete citations
