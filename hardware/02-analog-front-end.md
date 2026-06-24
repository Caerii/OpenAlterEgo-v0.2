# Analog front-end (AFE)

Selection and configuration of the biopotential acquisition chain: electrode → protection → PGA → ADC.

---

## Why ADS1299-class AFE

The ADS1299 family (TI) is the de facto choice for multichannel wearable biopotentials in both research and OpenAlterEgo's reference designs.

| Spec | ADS1299-8 | OpenAlterEgo use |
|------|-----------|------------------|
| Channels | 8 simultaneous | Matches Kapur 7–8 ch |
| Resolution | 24-bit ΔΣ | Kapur 2020 reports 24-bit ADC |
| Sample rates | **250 SPS – 16 kSPS** | Default **250 Hz**; optional 500–1000 Hz |
| PGA gain | 1, 2, 4, 6, 8, 12, **24** | AlterEgo: **24×**; wideband: **6–12×** |
| Input-referred noise | ~1 µV<sub>PP</sub> (70 Hz BW) | Sets SNR ceiling |
| BIAS / DRL | Integrated patient bias amp | Earlobe reference drive |
| Lead-off detect | Current source/sink | Dry electrode contact QC |
| Interface | SPI | nRF52840 SPI master |

**Datasheet:** [TI ADS1299](https://www.ti.com/lit/ds/symlink/ads1299.pdf)

**Also used in:** OpenBCI Cyton (AlterEgo 2018), BioGAP-Ultra / SilentWear (ADS1298, 16 ch) ([arXiv:2603.02847](https://arxiv.org/abs/2603.02847))

### Alternatives

| Part | When to consider |
|------|------------------|
| **ADS1298** | 8 ch, slightly different feature set; SilentWear dual-AFE layout |
| **ADS1294/1296** | 4–6 ch cost-reduced V1.5 |
| **MAX30001** | Single-lead ECG-focused; too few channels |
| **Discrete INA + SAR ADC** | Custom low-volume; more noise tuning burden |

---

## Sampling rate selection

| Rate | Literature | Use case | Nyquist limit |
|------|------------|----------|---------------|
| **250 Hz** | Kapur 2018, 2020 | AlterEgo envelope mode; **V0 default** | 125 Hz |
| **500 Hz** | Wang 2021, SilentWear | Wide DSP band (20–450 Hz) with margin | 250 Hz |
| **1000 Hz** | Tang 2025, Lai 2023, Gaddy 2020 | Full EMG bandwidth, SE-ResNet papers | 500 Hz |

### Decision rule for OpenAlterEgo

```
if preprocessing_mode == "standard" or "clinical":
    fs = 250 Hz          # AlterEgo-compatible envelope
elif preprocessing_mode == "wide":
    fs >= 500 Hz         # honest 20–450 Hz content (see sim/literature.py)
```

At **250 Hz**, analog content above ~125 Hz is physically absent — wide DSP mode still applies a 20 Hz high-pass mindset but cannot recover lost bandwidth. For wide-mode experiments, **upgrade fs to 500–1000 Hz** in V1 firmware.

**Firmware note:** ADS1299 supports 250 SPS natively; doubling rate is a register change with proportional SPI/BLE load.

---

## Gain (PGA) selection

EMG at the skin surface is typically **0.1–5 mV peak-to-peak** depending on muscle and electrode contact ([Merletti & Parker, EMG textbook](https://www.robertomerletti.it/)).

| Gain | Full-scale input (Vref=4.5 V) | Literature |
|------|-------------------------------|------------|
| **24×** | ~±94 mV | Kapur 2018 (OpenBCI 24×) |
| **12×** | ~±188 mV | Compromise for motion artifacts |
| **6×** | ~±375 mV | SilentWear @ 500 Hz ([arXiv:2603.02847](https://arxiv.org/abs/2603.02847)) |

**Recommendation:**
- **V0:** gain **24**, accept occasional clipping during bad contact; monitor in software
- **V1 dry electrodes:** start **12×**; higher offset/drift with dry contacts ([Sensors 2023.16](https://www.oaepublish.com/articles/ss.2023.16))
- Clip detection: flag windows where ADC saturates > 1% samples

### Host scaling (`AfeSpec`)

Default in `acquisition/packet.py`:

```python
AfeSpec(adc_bits=16, vref_v=2.4, gain=24.0)
```

Firmware may use 24-bit internal resolution but **quantize to int16** on the wire; `AfeSpec` on the host must match firmware gain/Vref for correct µV conversion.

---

## Noise budget

Target: **≥ 18 dB SNR static** ([Tang 2025](https://arxiv.org/abs/2504.13921)).

| Noise source | Mitigation | Reference |
|--------------|------------|-----------|
| 50/60 Hz mains | Notch + harmonic notch in DSP; twisted pair leads; BIAS | Kapur 2020 harmonic notch |
| Motion / cable sway | Secure routing; high-pass; mechanical design | Tang 2025: 33% SNR drop |
| Electrode contact | Wet gel (V0); conformal dry structure (V1+) | [PMC9460933](https://pmc.ncbi.nlm.nih.gov/articles/PMC9460933/) |
| ADC quantization | 24-bit ΔΣ; avoid gain too high → clipping | ADS1299 datasheet |
| BLE dropouts | OA v1 `sample_index` gap detect | `acquisition/ble_client.py` |
| MCU / digital switching | Separate analog LDO; star ground; keep digital away from analog inputs | TI layout guide |

### SNR measurement (align with software)

Use `openalterego/dsp/quality.py` definitions on acquired data:
- Signal band: 20–450 Hz (wide) or 1–50 Hz (standard)
- Noise band: high-frequency residual or inter-trial baseline

---

## Input protection

Minimum network per electrode lead (before AFE):

```
Lead ── [R_series 100kΩ–1MΩ] ── [ESD diode pair to rails] ── AFE input
```

**Purpose:**
- Limit patient leakage current under fault ([IEC 60601-1](https://webstore.iec.ch/) mindset)
- Survive ESD during handling
- Limit current if electrode shorted

**Patent reference:** US10878818B2 discusses isolation and patient safety for wearable biopotential systems.

---

## BIAS / driven-right-leg (DRL)

ADS1299 integrates a patient bias amplifier that drives a reference electrode to cancel common-mode interference.

| Configuration | Literature |
|---------------|------------|
| Reference on **earlobe** (unipolar ref) | Kapur 2020 — 4 face + 4 neck + earlobe ref/bias |
| Reference on **wrist** | Kapur 2018 |
| **Fully differential**, no wet ref | SilentWear — shorted ground at back of neck ([arXiv:2603.02847](https://arxiv.org/abs/2603.02847)) |

**OpenAlterEgo V0 recommendation:** Earlobe reference + bias (AlterEgo 2020 clinical layout) — simplest path to good CMRR with gel electrodes.

**V2 dry neckband:** Evaluate overlapping differential pairs (SilentWear) to eliminate gel reference.

---

## Channel count vs accuracy

| Channels | Accuracy example | Source |
|----------|------------------|--------|
| 3 | 85.9% (26 NATO) | Lai 2023 |
| 4 | 92.6% (110 words), 96% (10 cmds) | Wang 2021; Tang 2025 |
| 7–8 | ~92% (AlterEgo vocab) | Kapur 2018 |
| 10 (optimized from HD) | > 86% | Ji 2021 |
| 22 | Research / phoneme | Gowda 2024 |

**Hardware implication:** **8-ch AFE is sufficient** for command vocabulary. Do not block V0 on 16+ channels.

---

## Anti-aliasing

ADS1299 integrates sinc filters; usable bandwidth is ~0.3 × data rate. At 250 SPS, analog bandwidth is ~75 Hz — consistent with AlterEgo envelope acquisition. For 1000 SPS, bandwidth extends to ~300 Hz; DSP wide band (20–450 Hz) still needs fs ≥ 920 Hz for full content ([`sim/literature.py`](../software/python/openalterego/sim/literature.py)).

---

## V0 bring-up checklist

1. Configure ADS1299: 250 SPS, gain 24, internal reference, BIAS enabled
2. Verify SPI readback and channel count
3. Short inputs → measure input-referred noise floor
4. Apply sine calibration signal → verify µV scaling end-to-end
5. Stream OA v1 packets → `openalterego collect ble` → check `sample_index` continuity
6. Attach electrodes → measure per-channel SNR during rest and silent speech

See [04-ble-firmware-protocol.md](04-ble-firmware-protocol.md) for firmware interface.
