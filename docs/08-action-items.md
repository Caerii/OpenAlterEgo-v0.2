# Action Items: Research Validation Findings

**Last updated:** June 2026 — reflects implemented stack. Open items only.

Quick reference. Full plan: [`14-systematic-roadmap.md`](14-systematic-roadmap.md). Bibliography: [`12-references.md`](12-references.md).

---

## ✅ Resolved (was critical)

| Item | Resolution |
|------|------------|
| Per-user personalization | `users/*`, calibrate, train, serve |
| Fixed confidence threshold | Per-user calibration + adaptive EMA in `runtime/streaming.py` |
| Clinical preprocessing | `PreprocessingMode.clinical` in `dsp/filters.py` |
| Wide bandpass | `PreprocessingMode.wide` (20–450 Hz) |
| Harmonic notching | `apply_notch_with_harmonics()` |
| Motion / SNR monitoring | `dsp/quality.py`, online monitor, calibrate warnings |
| User CLI | `openalterego user`, `calibrate`, `collect` |
| Configurable window/stride | `UserProfile`, serve/train flags |

---

## 🔴 Open — Phase A (validation)

| # | Action | Files |
|---|--------|-------|
| A1 | A/B **standard vs wide** on real session data; publish numbers | eval script, `docs/` |
| A2 | **Latency benchmark** (p50/p95 acquisition→token) | `api/server.py`, script |
| A3 | **BLE events.csv** labeling workflow after `collect ble` | `users/collect.py`, `USER_GUIDE.md` |
| A4 | Import adapter for **Gowda OSF** or Gaddy data smoke test | `ml/` |

---

## 🟡 Open — Phase B/C (hardware & robustness)

| # | Action | Files |
|---|--------|-------|
| B1 | V0 hardware bring-up (OpenBCI / ADS1299 + BLE UUIDs) | `acquisition/ble_client.py`, `hardware/` |
| B2 | Electrode placement guide with anatomy diagrams | `docs/` |
| C1 | Optional **motion gating** in online preprocess | `dsp/online.py` |
| C2 | **Per-channel SNR** + weak-channel meta in serve | `dsp/quality.py`, `api/server.py` |
| C3 | **Re-calibration hint** when live SNR drops vs `baseline_snr` | `users/profile.py`, serve |
| C4 | Window sweep 600 vs 1500 ms — accuracy vs latency | `runtime/streaming.py` |

---

## 🟢 Open — Phase D/E (research / future)

| # | Action | Literature |
|---|--------|------------|
| D1 | SE-ResNet or channel-attention blocks | Tang 2025 |
| D2 | Knowledge distillation ensemble → student | Lai 2023 |
| D3 | SpeechNet-scale tiny CNN | Meier 2025 |
| D4 | Channel importance visualization | Gowda 2024 |
| E1 | Phoneme / Seq2Seq / CTC decoder | Gaddy 2020, Gowda 2025 |
| E2 | LLM or LM post-processing | MONA LISA 2024 |
| E3 | SPD / Riemannian representations | Gowda 2024 |

---

## Documentation updates needed

- [ ] Trim stale paragraphs in `TODO.md` (duplicate of `USER_GUIDE.md`)
- [ ] Add electrode placement doc (reference Deng 2023, MDPI 2025)
- [ ] Add latency budget doc with measured p50/p95
- [ ] Literature comparison table in eval results note

---

## Success criteria (unchanged)

- Per-user accuracy ≥ **85%** on held-out calibration data
- End-to-end latency < **500 ms**
- Static SNR ≥ **18 dB**; warn when motion SNR < **12 dB**
- 50–200 samples per token for calibration
