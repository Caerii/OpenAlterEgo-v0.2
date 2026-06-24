# Implementation Roadmap: Research-Validated Priorities

This document provides a quick-start guide to understanding what needs to be implemented, prioritized by criticality based on recent literature analysis.

**Quick Links:**
- 🗺️ **Systematic roadmap:** `14-systematic-roadmap.md` ← start here
- 🎤 **Open vocab + sim2real:** [`19-open-vocab-and-sim2real.md`](19-open-vocab-and-sim2real.md)
- 🧠 **Gowda / emg2speech:** [`gowda/00-README.md`](gowda/00-README.md) (paper, data, legal, validation)
- 🔴 **Critical Issues:** `11-priority-changes.md`
- 📚 **References:** [`12-references.md`](12-references.md) · **[`literature/README.md`](literature/README.md)** (master index + offline papers)
- ✅ **Full TODO:** `TODO.md`
- 📊 **Progress:** `10-implementation-progress.md`

---

## 🔴 CRITICAL: Three Issues That Must Be Fixed

### 1. Bandpass Filter Range Discrepancy (HIGHEST PRIORITY)

**The Problem:**
- Our system: **1-50 Hz** bandpass (based on AlterEgo 2018)
- Recent literature (2021-2024): **20-450/500 Hz** bandpass
- **Impact:** We may be filtering out critical signal content above 50 Hz

**Evidence:**
- Wang et al. (2021): 20-500 Hz → 92.64% accuracy on 110 words
- Tang et al. (2024): 20-450 Hz → 96% accuracy on 10 words
- Lai et al. (2023): 20-400 Hz → 85.9% accuracy on 26 words

**Solution:**
- Add "wide" preprocessing mode (20-450 Hz)
- Keep "standard" mode (1-50 Hz) for AlterEgo compatibility
- Test both modes and document when to use each

**See:** `11-priority-changes.md` Section 1

---

### 2. Motion Artifact Handling (CRITICAL)

**The Problem:**
- Motion artifacts reduce SNR by **33%** (18.9 dB → 12.7 dB)
- We don't detect or handle motion artifacts
- Real-world usage involves head movement, jaw motion, etc.

**Evidence:**
- Tang et al. (2024): Static SNR 18.9 dB, Motion SNR 12.7 dB
- Motion causes low-frequency drift and impedance variations
- 4th order Butterworth 20-450 Hz effectively suppresses artifacts

**Solution:**
- Add motion artifact detection (low-frequency drift monitoring)
- Improve high-pass filtering (< 20 Hz removal)
- Add signal quality monitoring (real-time SNR)
- Include in calibration workflow

**See:** `11-priority-changes.md` Section 2

---

### 3. Per-User Personalization System (CRITICAL)

**The Problem:**
- We have user management infrastructure but no calibration workflow
- Cross-user generalization is poor without personalization
- Each user needs their own model and threshold

**Evidence:**
- Gowda et al. (2024): Domain shift = "change of basis" - eigenbasis vectors differ per person
- All papers show per-user/subject training is essential
- Geometry paper explains why: anatomical, physiological, and neural differences

**Solution:**
- Implement calibration workflow (Phase 3)
- Per-user model training (Phase 5)
- Per-user threshold computation (Phase 4)
- User-aware serving (Phase 6)

**See:** `TODO.md` Phases 3-6

---

## 📋 Implementation Status (June 2026)

### ✅ Completed (Phases 1–6 + critical literature fixes)
- Standard / wide / clinical preprocessing; harmonic notching
- User profiles, calibration, collection (`collect sim|ble`)
- User-aware train and serve; adaptive threshold + SNR gating
- Motion / SNR quality monitoring; literature-aligned simulation
- 65+ pytest tests; `USER_GUIDE.md`

### 🚧 Next (Phase A — validation on real data)
- A/B standard vs wide bandpass on human EMG
- Latency benchmarks; BLE labeling workflow
- External dataset compatibility (Gowda OSF, Gaddy)

See **`14-systematic-roadmap.md`** for the full phased plan.

---

## 🎯 Recommended Implementation Order

### Week 1-2: Critical Fixes
1. **Add "wide" bandpass mode** (20-450 Hz)
   - Extend `PreprocessingMode` enum
   - Update `get_filter_spec_for_mode()`
   - Test and compare with "standard" mode
   - Document when to use each

2. **Motion artifact detection**
   - Add low-frequency drift monitoring
   - Improve high-pass filtering
   - Add SNR calculation
   - Integrate into calibration

### Week 3-4: Core Personalization
3. **Calibration workflow** (Phase 3)
   - Implement `calibrate_user()` function
   - Compute per-user thresholds
   - Train per-user models
   - Generate calibration reports

4. **User-aware training & serving** (Phases 5-6)
   - Integrate with training script
   - Integrate with server
   - CLI commands

### Week 5+: Enhancements
5. **Adaptive thresholding** (Phase 4)
6. **Signal quality monitoring**
7. **Documentation and guides**

---

## 📚 Key Papers & Why They Matter

### Foundation Papers
- **Kapur et al. (2018)**: Original AlterEgo - validates our 250 Hz, 1-50 Hz approach
- **Kapur et al. (2020)**: Clinical validation - validates 0.5-8 Hz clinical mode

### Critical Recent Papers
- **Wang et al. (2021)**: ⚠️ Uses 20-500 Hz - suggests our bandpass may be too narrow
- **Tang et al. (2024)**: ⚠️ Uses 20-450 Hz, shows motion artifact impact (33% SNR drop)
- **Gowda et al. (2024)**: ✅ Validates personalization ("change of basis" explanation)
- **Lai et al. (2023)**: ✅ Validates ResNet1D architecture, uses 20-400 Hz

**See:** `12-references.md` for complete list with citations and importance

---

## 🔍 Key Insights from Literature

### What We Got Right ✅
- **250 Hz sampling rate** - Validated by AlterEgo papers
- **1D CNN architecture** - Validated by multiple recent papers
- **Per-user personalization** - Validated by geometry paper
- **7-8 channels** - Provides redundancy (4 channels sufficient per literature)

### What Needs Attention ⚠️
- **Bandpass range** - Recent papers use 20-450 Hz vs. our 1-50 Hz
- **Motion artifacts** - Significant impact (33% SNR degradation)
- **Window size** - Literature uses 1.5-3s vs. our 600ms
- **Signal quality monitoring** - Not implemented but critical

### Future Enhancements 🟢
- SE-ResNet architecture (adaptive channel weighting)
- Knowledge distillation (model compression)
- SPD matrix representation (geometric priors)
- Multimodal integration (IMU, eye tracking)

---

## 📖 Documentation Structure

```
docs/
├── 00-README-IMPLEMENTATION.md  ← You are here (quick start)
├── 07-research-validation.md     ← Technical parameter validation
├── 08-action-items.md            ← Quick reference checklist
├── 09-implementation-design.md   ← Detailed architecture
├── 10-implementation-progress.md ← Current status
├── 11-priority-changes.md        ← 🔴 Critical issues analysis
├── 12-references.md              ← 📚 Complete reference list
└── TODO.md                        ← Full implementation plan
```

---

## 🚀 Getting Started

1. **Read `11-priority-changes.md`** - Understand the critical issues
2. **Review `12-references.md`** - Understand the research foundation
3. **Check `10-implementation-progress.md`** - See what's done
4. **Follow `TODO.md`** - Implement remaining work in priority order

**Start with:** Adding "wide" bandpass mode - this is the highest priority change that may significantly impact accuracy.

---

## Questions?

- **Why is bandpass range critical?** Recent papers consistently use 20-450 Hz and achieve 85-96% accuracy. Our 1-50 Hz may be missing important signal content.
- **Why is motion artifact handling critical?** Motion reduces SNR by 33%, which can significantly degrade accuracy in real-world use.
- **Why is personalization critical?** Geometry paper explains domain shift as "change of basis" - each person has different eigenbasis vectors, requiring per-user models.

**See the detailed analysis in `11-priority-changes.md` and `12-references.md` for complete explanations.**
