# Priority Changes: Critical Base Functionality

This document identifies the **highest priority changes** required for proper base functionality, based on analysis of recent literature and comparison with our current implementation.

**Status:** 🔴 Critical | 🟡 Important | 🟢 Enhancement

---

## 🔴 CRITICAL: Bandpass Filter Range Discrepancy

### Issue
**Our current implementation:** 1-50 Hz bandpass (for silent speech envelope)  
**Recent literature (2021-2024):** 20-450/500 Hz bandpass (for actual EMG signals)

### Evidence from Papers

1. **Wang et al. (2021)** - Tattoo-like electronics:
   - Uses **20-500 Hz** bandpass filter
   - Achieves **92.64% accuracy** on 110 words
   - States: "most useful information in sEMG signals is in the frequency band between 15Hz and 450Hz"

2. **Tang et al. (2024)** - Headphone-integrated SSI:
   - Uses **20-450 Hz** bandpass (4th order Butterworth)
   - Achieves **96% accuracy** on 10 words
   - Notes: "This frequency range is chosen based on the physiological properties of EMG signals, where most speech-related EMG activity lies between 20 Hz and 400 Hz"

3. **Lai et al. (2023)** - Knowledge Distillation:
   - Uses **20-400 Hz** bandpass (10th order Butterworth)
   - Achieves **85.9% accuracy** on 26 NATO alphabet
   - States: "most useful sEMG signals are located between 15-28 Hz to 400-450 Hz"

### Why This Matters

- **We may be filtering out critical signal content** above 50 Hz
- Literature consistently shows EMG activity extends to 400-500 Hz
- Our 1-50 Hz range was based on AlterEgo papers (2018, 2020), but newer work suggests wider bands are needed
- The discrepancy might explain why we need more channels (7-8) vs. literature (3-4 channels achieve similar accuracy)

### Required Action

**Option A: Add "wide" preprocessing mode (RECOMMENDED)**
- Keep existing 1-50 Hz as "standard" (for AlterEgo compatibility)
- Add new "wide" mode: 20-450 Hz (for modern EMG processing)
- Allow users to choose based on their hardware/signal characteristics

**Option B: Make bandpass fully configurable**
- Expose `--bandpass-low` and `--bandpass-high` parameters
- Default to 1-50 Hz (backward compatible)
- Document trade-offs

**Implementation:**
- [ ] Add "wide" mode to `PreprocessingMode`
- [ ] Update `get_filter_spec_for_mode()` to support "wide" (20-450 Hz)
- [ ] Update `OnlinePreprocessor` to support wider bandpass
- [ ] Add tests comparing "standard" vs "wide" modes
- [ ] Document when to use each mode

**Priority:** 🔴 **CRITICAL** - This likely impacts accuracy significantly

---

## 🔴 CRITICAL: Motion Artifact Handling

### Issue
Motion artifacts significantly degrade signal quality, but we don't have robust detection/handling.

### Evidence from Papers

**Tang et al. (2024)** - Headphone paper:
- Static conditions: **18.9 dB SNR**
- Motion conditions: **12.7 dB SNR** (33% degradation!)
- States: "motion-induced impedance variations at the electrode-skin interface" cause low-frequency artifacts
- Uses 4th order Butterworth 20-450 Hz to suppress artifacts below 20 Hz

**Wang et al. (2021)** - Tattoo paper:
- Tests long-term wear during exercise, dining, temperature changes
- Shows signal features remain stable with proper filtering
- Uses wavelet denoising + bandpass filtering

### Why This Matters

- Real-world usage involves head movement, jaw motion, facial expressions
- Motion artifacts can reduce accuracy by 20-30%
- Without detection, users won't know when signal quality degrades
- Calibration should include motion artifact assessment

### Required Action

- [ ] Add motion artifact detection:
  - Monitor low-frequency drift (< 5 Hz)
  - Track baseline wandering
  - Calculate motion artifact index
- [ ] Improve filtering:
  - Ensure high-pass filter effectively removes < 20 Hz artifacts
  - Consider adaptive filtering based on motion state
- [ ] Add signal quality monitoring:
  - Real-time SNR calculation
  - Alert when signal quality degrades
  - Suggest re-calibration if needed
- [ ] Include in calibration:
  - Test under motion conditions
  - Set baseline SNR thresholds
  - Document motion tolerance

**Priority:** 🔴 **CRITICAL** - Essential for real-world deployment

---

## 🔴 CRITICAL: Per-User Personalization System

### Issue
We have user management infrastructure but no calibration workflow.

### Evidence from Papers

**Gowda et al. (2024)** - Geometry paper:
- **Key finding**: "Domain shift in sEMG signals due to combined effect of anatomical, physiological, and neural drive properties is characterized by a change of basis"
- Shows eigenbasis vectors differ per individual
- Validates that per-user training is essential
- States: "signals from different individuals have different eigenbasis vectors"

**All papers** show per-user or per-subject training:
- Tattoo paper: Per-user models
- Headphone paper: Per-subject evaluation
- Knowledge Distillation: Per-subject training

### Why This Matters

- Cross-user generalization is poor without personalization
- Each user needs their own model and threshold
- Calibration workflow is the foundation for everything else
- Without this, the system won't work reliably for real users

### Required Action

- [ ] Implement calibration workflow (Phase 3)
- [ ] Compute per-user confidence thresholds
- [ ] Train per-user models
- [ ] Store calibration metadata in UserProfile

**Priority:** 🔴 **CRITICAL** - Core functionality, blocks other features

---

## 🟡 IMPORTANT: Adaptive Channel Weighting

### Issue
We don't adapt to variable channel quality (some channels may be noisy).

### Evidence from Papers

**Tang et al. (2024)** - Headphone paper:
- Uses **1D SE-ResNet** with Squeeze-and-Excitation blocks
- Dynamically adjusts per-channel attention weights
- Assigns higher importance to well-coupled channels
- Suppresses noisy or unstable channels
- Shows this is critical for dry electrodes (variable contact)

### Why This Matters

- Electrode contact quality varies over time
- Some channels may be noisier than others
- Adaptive weighting improves robustness
- Especially important for wearable/dry electrode systems

### Required Action

- [ ] Consider adding SE blocks to our CNN (future enhancement)
- [ ] For now: Implement channel quality monitoring
- [ ] Track per-channel SNR
- [ ] Allow disabling noisy channels
- [ ] Document channel importance analysis

**Priority:** 🟡 **IMPORTANT** - Improves robustness but not blocking

---

## 🟡 IMPORTANT: Window Size Optimization

### Issue
Our 600ms window may be suboptimal compared to literature.

### Evidence from Papers

- **Gowda et al. (2024)**: Uses 1.5s for words, 100-400ms sliding windows
- **Wang et al. (2021)**: Uses 2000ms total (800ms before + 1200ms after trigger)
- **Tang et al. (2024)**: Uses 3000ms (3-second windows)
- **Lai et al. (2023)**: Uses 1500ms (1.5s windows)

### Why This Matters

- Longer windows capture more context
- May improve accuracy at cost of latency
- Should be user-configurable
- May need different windows for different use cases

### Required Action

- [ ] Make window size configurable (already done, but validate)
- [ ] Test longer windows (1-2s) for accuracy improvement
- [ ] Document latency vs. accuracy trade-off
- [ ] Add to UserProfile for per-user optimization

**Priority:** 🟡 **IMPORTANT** - Can improve accuracy but not critical

---

## 🟢 ENHANCEMENT: Advanced Architectures

### Future Work (Not Blocking)

- SE-ResNet architecture (adaptive channel weighting)
- Knowledge distillation (ensemble → lightweight model)
- SPD matrix representation (Riemannian geometry)
- Multimodal integration (IMU, eye tracking)

**Priority:** 🟢 **ENHANCEMENT** - Nice to have, not required for MVP

---

## Summary: Critical Path to Base Functionality

### Must Have (Blocking):
1. ✅ **Per-User Personalization** (Phase 3-6) - Core feature
2. 🔴 **Bandpass Filter Range** - May be missing signal content
3. 🔴 **Motion Artifact Handling** - Essential for real-world use

### Should Have (Important):
4. 🟡 **Adaptive Thresholding** (Phase 4) - Improves accuracy
5. 🟡 **Window Size Optimization** - Can improve accuracy
6. 🟡 **Signal Quality Monitoring** - User feedback

### Nice to Have (Enhancement):
7. 🟢 **Advanced Architectures** - Future research
8. 🟢 **Geometric Priors** - Advanced topic
9. 🟢 **Multimodal Integration** - Future work

---

## Implementation Order

1. **Week 1-2:** Per-user personalization (Phases 3-6) - Core functionality
2. **Week 3:** Bandpass filter "wide" mode - Critical accuracy fix
3. **Week 4:** Motion artifact detection and handling - Real-world robustness
4. **Week 5+:** Adaptive thresholding, window optimization, etc.

---

## Testing Priorities

- [ ] Compare "standard" (1-50 Hz) vs "wide" (20-450 Hz) modes on same data
- [ ] Test motion artifact detection and filtering
- [ ] Validate per-user calibration improves accuracy
- [ ] Benchmark latency with different window sizes
- [ ] Test signal quality monitoring in real-world conditions
