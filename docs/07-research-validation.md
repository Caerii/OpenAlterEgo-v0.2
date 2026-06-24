# Research Validation: Technical Parameters & Assumptions

This document validates OpenAlterEgo-v0.2's technical parameters against published research and identifies required changes.

## Executive Summary

**Status:** Core architecture is sound, but several critical parameters need validation and adjustment based on research findings.

**Key Findings:**
- ✅ 250 Hz sampling rate is **validated** by AlterEgo papers (intentional for silent speech)
- ⚠️ Bandpass 1-50 Hz is **too wide** - research suggests 0.5-8 Hz for clinical cases
- ✅ 600ms window size is **reasonable** for speech patterns
- ⚠️ 120ms stride needs **empirical validation** (no literature specification)
- ⚠️ 0.70 confidence threshold is **untested** - needs user-specific calibration
- ❌ Missing **per-user personalization** (critical gap)

---

## 1. Sampling Rate: 250 Hz

### Current Implementation
- Default: **250 Hz** throughout codebase
- Used in: `SimStreamConfig`, `BleSpec`, `VirtualBleSpec`, training scripts

### Research Validation

**AlterEgo Papers (2018, 2020):**
- Explicitly report **250 Hz sampling rate** for silent speech EMG
- This is **intentionally lower** than standard EMG (typically 1-10 kHz)

**Why 250 Hz works for silent speech:**
- Silent speech EMG has **lower frequency content** than standard muscle EMG
- Subvocalization produces slower, more sustained activations
- 250 Hz provides ~125 Hz Nyquist frequency, sufficient for 0-50 Hz bandpass

**Standard EMG Context:**
- Typical EMG: 1-10 kHz (2 kHz common)
- Frequency content: 20-500 Hz
- Silent speech is fundamentally different - lower frequency, more envelope-like

### Verdict: ✅ **VALIDATED**
- 250 Hz is correct for silent speech applications
- Matches AlterEgo research exactly
- **No changes needed**

---

## 2. Bandpass Filter: 1.0-50.0 Hz

### Current Implementation
- Default: **1.0-50.0 Hz** (4th order Butterworth)
- Location: `dsp/filters.py`, `dsp/online.py`
- Alternative mentioned in docs: 0.5-8 Hz for clinical cases

### Research Validation

**AlterEgo 2018 Paper:**
- Reports: **1.3-50 Hz** bandpass (4th order Butterworth)
- ✅ Your 1.0-50.0 Hz is **very close** and appropriate

**AlterEgo 2020 Paper (Clinical MS patients):**
- Reports: **0.5-8 Hz** bandpass
- This is **much narrower** and lower frequency
- Used specifically for dysphonic patients

**Standard EMG:**
- Typical: 20-500 Hz (completely different application)
- Not relevant for silent speech

### Issues Identified

1. **Documentation mentions both ranges but code only implements 1-50 Hz**
   - Need to support both preprocessing pipelines
   - Clinical mode should use 0.5-8 Hz

2. **High-pass cutoff (1.0 Hz) may be too low**
   - 2020 paper uses 0.5 Hz, but also mentions ">0.5 Hz" high-pass
   - Very low frequencies contain drift/motion artifacts
   - Consider 0.5-1.0 Hz as minimum

### Required Changes

1. **Add clinical preprocessing mode:**
   ```python
   # In dsp/filters.py, add:
   def preprocess_clinical(x, fs_hz, ...):
       # 0.5-8 Hz bandpass as per 2020 paper
   ```

2. **Make bandpass configurable per use case:**
   - Default: 1.0-50.0 Hz (general use)
   - Clinical: 0.5-8.0 Hz (dysphonia/MS patients)
   - Expose via CLI/config

3. **Document the trade-off:**
   - Wider band (1-50 Hz): More information, more noise
   - Narrower band (0.5-8 Hz): Less noise, may miss some signal content

### Verdict: ⚠️ **NEEDS ADJUSTMENT**
- Current 1-50 Hz is valid for general use
- Must add 0.5-8 Hz option for clinical cases
- Consider making high-pass configurable (0.5-1.0 Hz range)

---

## 3. Window Size: 600 ms

### Current Implementation
- Default: **600 ms** for training and inference
- Location: `runtime/streaming.py`, `ml/train.py`
- Converts to samples: `600 ms * 250 Hz = 150 samples`

### Research Validation

**Literature Findings:**
- No explicit window size specified in AlterEgo papers
- Speech phonemes typically: 50-200 ms
- Words: 200-800 ms
- 600 ms captures **1-3 phonemes or 1 word**

**EMG Classification Research:**
- Typical EMG windows: 200-1000 ms
- Shorter windows (200-400 ms): Lower latency, less context
- Longer windows (800-1000 ms): More context, higher latency

**Real-time Considerations:**
- 600 ms provides good balance
- Allows capture of complete subvocalization events
- Not too long to cause noticeable latency

### Potential Issues

1. **Fixed window may not match actual event duration**
   - Subvocalization varies: 200-900 ms (from simulation config)
   - Some events may be shorter than 600 ms
   - Padding/cropping handles this, but may lose information

2. **Latency concerns:**
   - 600 ms window + processing = ~700-800 ms total latency
   - May feel sluggish for rapid commands
   - Consider shorter windows (300-400 ms) for low-latency mode

### Required Changes

1. **Make window size configurable:**
   - Default: 600 ms (balanced)
   - Low-latency: 300-400 ms (faster response)
   - High-accuracy: 800-1000 ms (more context)

2. **Validate with real data:**
   - Measure actual subvocalization durations
   - Optimize window size based on user data

### Verdict: ✅ **REASONABLE, BUT MAKE CONFIGURABLE**
- 600 ms is a good default
- Should be user-configurable
- Consider shorter windows for low-latency applications

---

## 4. Inference Stride: 120 ms

### Current Implementation
- Default: **120 ms** stride for sliding window
- Location: `runtime/streaming.py`
- Converts to: `120 ms * 250 Hz = 30 samples`

### Research Validation

**Literature Findings:**
- **No explicit stride specified in AlterEgo papers**
- This is an engineering choice, not validated in research
- Needs empirical validation

**Real-time Processing Considerations:**
- 120 ms = 8.3 predictions per second
- Reasonable for command interfaces
- May miss rapid transitions

**Typical EMG Processing:**
- Overlap: 50-75% common
- 120 ms stride on 600 ms window = 80% overlap
- This is reasonable for smooth predictions

### Potential Issues

1. **No research validation**
   - Must test empirically
   - May need adjustment based on user feedback

2. **May be too conservative**
   - 120 ms = noticeable delay
   - Consider 50-100 ms for more responsive feel

3. **Computational cost**
   - More frequent predictions = higher CPU usage
   - Balance responsiveness vs. compute

### Required Changes

1. **Add latency benchmarking:**
   - Measure end-to-end latency (acquisition → prediction → output)
   - Target: <500 ms for natural interaction

2. **Make stride configurable:**
   - Default: 120 ms (balanced)
   - Responsive: 50-80 ms (lower latency)
   - Efficient: 150-200 ms (lower compute)

3. **Validate empirically:**
   - Test with real users
   - Measure false positive/negative rates vs. stride

### Verdict: ⚠️ **NEEDS EMPIRICAL VALIDATION**
- 120 ms is reasonable but unvalidated
- Should be configurable
- Requires user testing to optimize

---

## 5. Confidence Threshold: 0.70

### Current Implementation
- Default: **0.70** (70%) minimum confidence
- Location: `runtime/streaming.py`
- Applied after debouncing (stable N predictions)

### Research Validation

**Literature Findings:**
- **No explicit confidence threshold in AlterEgo papers**
- This is an implementation choice
- 92% accuracy reported, but threshold not specified

**Typical ML Confidence Thresholds:**
- High-stakes: 0.90-0.95 (fewer false positives)
- Balanced: 0.70-0.80 (common default)
- Permissive: 0.50-0.60 (more predictions, more errors)

### Critical Issues

1. **Not user-specific**
   - Different users have different signal quality
   - Some users may need 0.85, others 0.60
   - **Must be per-user adaptive**

2. **Fixed threshold is problematic**
   - Signal quality varies over time (electrode drift, sweat)
   - Should adapt to current signal conditions
   - Consider dynamic thresholding

3. **No rejection mechanism**
   - What happens below threshold? (silence vs. low-confidence prediction)
   - Should explicitly handle "unknown" class

### Required Changes

1. **Implement per-user threshold calibration:**
   ```python
   # During user enrollment:
   - Collect baseline data
   - Measure typical confidence for known tokens
   - Set user-specific threshold (e.g., mean - 2*std)
   ```

2. **Add adaptive thresholding:**
   - Monitor signal quality metrics
   - Adjust threshold based on current conditions
   - Alert when signal degrades

3. **Add explicit rejection class:**
   - Below threshold → emit "<unknown>" or silence
   - Don't emit low-confidence predictions

4. **Make threshold configurable:**
   - Per-user calibration sets default
   - User can adjust sensitivity
   - Document trade-offs (precision vs. recall)

### Verdict: ❌ **CRITICAL GAP - NEEDS MAJOR CHANGES**
- 0.70 is arbitrary and not validated
- Must be per-user adaptive
- Requires calibration pipeline

---

## 6. Per-User Personalization (MISSING)

### Current Implementation
- **No explicit per-user personalization**
- Training script assumes single dataset
- No user enrollment/calibration workflow

### Research Validation

**AlterEgo Papers:**
- Title explicitly says **"Personalized"**
- Per-user training is **critical** for performance
- 92% accuracy is per-user, not cross-user

**Why Personalization Matters:**
- EMG signals are highly individual
- Muscle activation patterns vary significantly
- Electrode placement affects signal characteristics
- Cross-user generalization is poor

### Critical Gap

**Current State:**
- Training assumes single user dataset
- No user ID tracking
- No per-user model storage
- No calibration workflow

**What's Missing:**
1. User enrollment system
2. Per-user data collection protocol
3. Per-user model training/fine-tuning
4. User-specific threshold calibration
5. Model versioning per user

### Required Changes

1. **Add user management:**
   ```python
   # New module: openalterego/users.py
   - User registration
   - Per-user data directories
   - Per-user model storage
   ```

2. **Calibration workflow:**
   ```python
   # New command: openalterego calibrate --user-id <id>
   - Guided data collection
   - Minimum sample requirements
   - Quality checks
   - Model training
   ```

3. **Per-user inference:**
   ```python
   # Server loads user-specific model
   openalterego serve --user-id <id> --model ./users/<id>/model.pt
   ```

4. **Document calibration requirements:**
   - Minimum samples per token (e.g., 50-200)
   - Expected calibration time (e.g., 10-30 minutes)
   - Re-calibration triggers (electrode shift, low accuracy)

### Verdict: ❌ **CRITICAL MISSING FEATURE**
- Personalization is core to AlterEgo approach
- Must implement before production use
- This is the highest priority gap

---

## 7. Notch Filter: 60 Hz

### Current Implementation
- Default: **60 Hz** notch filter (Q=30)
- Location: `dsp/filters.py`, `dsp/online.py`
- Configurable but defaults to 60 Hz

### Research Validation

**AlterEgo Papers:**
- 2018: Reports 60 Hz notch
- 2020: Reports 60 Hz and harmonics

**Power Line Interference:**
- North America: 60 Hz
- Europe/Asia: 50 Hz
- Harmonics: 120 Hz, 180 Hz (60 Hz) or 100 Hz, 150 Hz (50 Hz)

### Issues Identified

1. **Hardcoded to 60 Hz**
   - Should detect/configure based on region
   - Or support both 50/60 Hz

2. **Harmonics not addressed**
   - 2020 paper mentions harmonics
   - Current implementation only notches fundamental

### Required Changes

1. **Make notch frequency configurable:**
   ```python
   # Auto-detect or configurable
   notch_hz: Optional[float] = None  # None = auto-detect
   # Or explicit: 50, 60, or None
   ```

2. **Add harmonic notching (optional):**
   ```python
   # For 60 Hz: also notch 120, 180 Hz
   # For 50 Hz: also notch 100, 150 Hz
   ```

### Verdict: ⚠️ **NEEDS IMPROVEMENT**
- 60 Hz default is correct for North America
- Should be configurable for international use
- Consider harmonic notching for better noise rejection

---

## 8. Channel Count: 7-8 Channels

### Current Implementation
- Default: **8 channels**
- Configurable: 7-8 mentioned in docs
- Location: Throughout codebase

### Research Validation

**AlterEgo Papers:**
- 2018: **7 channels** (specific electrode locations)
- 2020: **8 channels** (4 face + 4 neck)
- Both approaches validated

**Electrode Placement:**
- Face: orbicularis oris, levator anguli oris, mentum
- Neck: hyoid, laryngeal, platysma, digastric
- Reference: wrist or earlobe

### Verdict: ✅ **VALIDATED**
- 7-8 channels is correct
- Both configurations are research-validated
- **No changes needed** (but document placement clearly)

---

## 9. Model Architecture: 1D CNN

### Current Implementation
- Architecture: 1D CNN with 4 conv blocks
- Input: (channels, time) = (8, 150) for 600ms @ 250Hz
- Output: Softmax over vocabulary

### Research Validation

**AlterEgo Papers:**
- 2018: Uses 1D CNN (described in paper)
- Architecture details: Conv+Pool blocks, FC layers
- Your implementation matches this approach

**Typical EMG CNNs:**
- 1D CNN is standard for time series EMG
- Your architecture is reasonable
- May need tuning based on actual data

### Verdict: ✅ **APPROPRIATE**
- 1D CNN is correct choice
- Matches AlterEgo approach
- Architecture details may need tuning with real data

---

## Summary of Required Changes

### Critical (Must Fix)

1. **Per-User Personalization System**
   - Priority: **HIGHEST**
   - Add user management, calibration workflow, per-user models
   - This is core to AlterEgo approach

2. **Adaptive Confidence Thresholding**
   - Priority: **HIGH**
   - Per-user calibration
   - Dynamic adjustment based on signal quality

### Important (Should Fix)

3. **Clinical Preprocessing Mode**
   - Add 0.5-8 Hz bandpass option
   - Make bandpass configurable

4. **Configurable Parameters**
   - Window size (300-1000 ms range)
   - Stride (50-200 ms range)
   - Make defaults but allow tuning

5. **Notch Filter Improvements**
   - Support 50/60 Hz (auto-detect or config)
   - Optional harmonic notching

### Nice to Have

6. **Latency Benchmarking**
   - Measure end-to-end latency
   - Document performance targets

7. **Signal Quality Monitoring**
   - Detect electrode drift
   - Alert when re-calibration needed

8. **Rejection Handling**
   - Explicit "<unknown>" class
   - Don't emit low-confidence predictions

---

## Validation Status Matrix

| Parameter | Current Value | Research Status | Action Required |
|-----------|---------------|-----------------|-----------------|
| Sampling Rate | 250 Hz | ✅ Validated | None |
| Bandpass | 1-50 Hz | ⚠️ Partial | Add 0.5-8 Hz option |
| Window Size | 600 ms | ✅ Reasonable | Make configurable |
| Stride | 120 ms | ⚠️ Unvalidated | Test & make configurable |
| Confidence | 0.70 | ❌ Arbitrary | Per-user calibration |
| Notch | 60 Hz | ⚠️ Partial | Support 50 Hz, harmonics |
| Channels | 7-8 | ✅ Validated | None |
| Model | 1D CNN | ✅ Appropriate | Tune with real data |
| Personalization | Missing | ❌ Critical | Implement system |

---

## Next Steps

1. **Immediate:** Implement per-user personalization system
2. **Short-term:** Add clinical preprocessing mode, configurable parameters
3. **Medium-term:** Empirical validation with real EMG data
4. **Long-term:** Signal quality monitoring, adaptive thresholds

---

## References

- Kapur, Kapur, Maes (2018) - "AlterEgo: A Personalized Wearable Silent Speech Interface" (IUI 2018)
- Kapur et al. (2020) - "Non-Invasive Silent Speech Recognition in Dysphonic Multiple Sclerosis" (PMLR/ML4H)
- US20190074012A1 / US10878818B2 - "Methods and Apparatus for Silent Speech Interface" (patent)
