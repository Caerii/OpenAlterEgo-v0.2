# Architecture tiers: V0 → V1 → V2

Three hardware tiers with explicit **exit criteria** so software and mechanical work stay decoupled.

---

## Tier comparison

| | **V0 Benchtop** | **V1 Wearable PCB** | **V2 Comfort + repeatability** |
|---|-----------------|---------------------|----------------------------------|
| **Goal** | Prove signals + ML on real EMG | Untethered BLE wearable | Daily-wear repeatability |
| **Timeline** | Weeks | Months | Months+ |
| **AFE** | COTS dev kit | Custom PCB | V1 electronics in enclosure |
| **Electrodes** | Pre-gelled Ag/AgCl | Gel or early dry | Frame-integrated dry/textile |
| **Placement** | Manual each session | Manual, improving | Mechanical registration |
| **fs** | 250 Hz (match AlterEgo) | 250–500 Hz | Same |
| **Transport** | USB or BLE from dev kit | OA v1 BLE firmware | Same |
| **Power** | Bench battery; USB isolator if tethered | LiPo 150–500 mAh | Optimized for >4 h |
| **Risk focus** | SNR, placement skill | EMI, BLE throughput, battery | Motion artifacts, comfort |

---

## V0: Benchtop rig (start here)

### Why first

The hard problems in silent speech are **electrode contact** and **placement repeatability**, not PCB layout ([Kapur 2018](https://dl.acm.org/doi/10.1145/3172944.3172977); [Deng 2023](https://doi.org/10.1109/TIM.2023.3276540)). A COTS biopotential front-end lets you debug DSP, calibration, and ML while electrode strategy converges.

### Recommended platforms

| Platform | Channels | ADC | Default fs | Notes |
|--------|----------|-----|------------|-------|
| **OpenBCI Cyton** | 8 | 24-bit ADS1299 | 250 Hz | AlterEgo 2018 explicitly mentions OpenBCI; large community |
| **OpenBCI Ganglion** | 4 | 24-bit (4 ch) | 200 Hz | Lower cost; matches Wang 2021 4-ch minimum |
| **TI ADS1299 EEG FE** | 4–8 | 24-bit | 250–16k SPS | Reference design for V1 PCB |
| **BioGAP-Ultra** (research) | 16 diff | 2× ADS1298 | 500 Hz | SilentWear platform ([arXiv:2603.02847](https://arxiv.org/abs/2603.02847)) |

### V0 exit criteria

- [ ] **Static SNR ≥ 18 dB** on at least 6/8 channels during silent token rehearsal ([Tang 2025](https://arxiv.org/abs/2504.13921) static reference)
- [ ] **Per-user calibration accuracy ≥ 85%** on held-out session (6-command vocab)
- [ ] **A/B preprocessing:** standard (1–50 Hz) vs wide (20–450 Hz) documented on same session
- [ ] **OA v1 or raw_i16** stream parsed by `openalterego collect ble` without sample gaps > 1 packet
- [ ] Electrode montage documented with photos + muscle labels

### V0 BOM

See [BOM.md](BOM.md#v0-benchtop).

---

## V1: Wearable custom PCB

### Architecture

```
Electrodes → protection → ADS1299 (8ch) → SPI → nRF52840 → BLE
                              ↑
                         BIAS/DRL → reference electrode
Power: LiPo → charger (MCP73831-class) → 3.3 V digital + clean analog rail
```

### Design requirements (from literature + patent)

| Requirement | Source |
|-------------|--------|
| 7–8 differential channels | Kapur 2018, 2020 |
| 24-bit ADC class | Kapur 2020; ADS1299 |
| 250 Hz default SPS (configurable 500–1000) | Kapur 2018; Wang 2021; Tang 2025 |
| PGA gain 6–24× | Kapur 2018 (24×); SilentWear (6× @ 500 Hz) |
| BIAS/DRL electrode drive | ADS1299 app note; Kapur 2020 earlobe ref |
| BLE (not WiFi primary) | Kapur 2018; low power wearable |
| Input protection + series resistors | US10878818B2; IEC 60601 mindset |
| Battery-powered during wear | Safety checklist in [05-power-safety.md](05-power-safety.md) |

### MCU selection

| MCU | BLE | SPI to ADS1299 | Notes |
|-----|-----|----------------|-------|
| **nRF52840** | 5.0, good stack | Yes | Quiet, low power; common in wearables |
| **nRF5340** | 5.x | Yes | Used in BioGAP-Ultra / SilentWear |
| **ESP32-S3** | WiFi + BLE | Yes | Tang 2025 headphone SSI; higher power |

**Recommendation:** nRF52840 for V1 (power, ecosystem, SoftDevice/Zephyr BLE).

### V1 exit criteria

- [ ] Untethered operation ≥ **2 hours** continuous streaming @ 250 Hz, 8 ch
- [ ] BLE packet loss **< 1%** at 1 m LOS (measured via `sample_index` gaps)
- [ ] OA v1 firmware passes `tests/test_packet.py` round-trip
- [ ] Motion SNR floor ≥ **12 dB** (warn below; [Tang 2025](https://arxiv.org/abs/2504.13921))
- [ ] Enclosure + strain relief; no tug on electrodes

### V1 BOM

See [BOM.md](BOM.md#v1-wearable-pcb).

---

## V2: Mechanical repeatability + daily wear

### Form-factor options (literature map)

| Form factor | Example | Channels | Electrode type | Trade-off |
|-------------|---------|----------|----------------|-----------|
| **Head band + arms** | AlterEgo 2018 | 7 | Gold/silver + paste or dry | High face coverage; visible |
| **Neckband** | SilentWear 2025 | 10–14 diff | Dry textile snaps | Discreet; neck-weighted ([Ji 2021](https://doi.org/10.1088/1741-2552/abca14)) |
| **Headphone earmuff** | Tang 2025 | 4 textile | Graphene/PEDOT:PSS dry | Very discreet; fewer channels |
| **Tattoo / patch** | Wang 2021 | 4 | Tattoo-like flexible | Long wear; specialized fab |

**OpenAlterEgo default path:** Start with **AlterEgo-style head band** (matches 7–8 ch software assumptions), evaluate **neckband** if face electrodes are unacceptable.

### V2 design principles

From Kapur 2018 mechanical description + our [06-mechanical-wearable.md](06-mechanical-wearable.md):

1. **Rigid electrode arms, adjustable length** — repeatability over comfort-only flex
2. **Route cables along frame** — minimize pull on skin during jaw motion
3. **Spring compliance at tip only** — constant gentle pressure, not slop in arm angle
4. **Quick re-seat protocol** — user can reposition in < 2 min with landmarks

### V2 exit criteria

- [ ] **Inter-session placement variance** < 5 mm on landmark muscles (measured)
- [ ] **Multi-hour wear** (≥ 4 h) without skin irritation (gel) or pressure marks (dry)
- [ ] **Inter-session accuracy drop** < 10% without re-calibration (target from SilentWear inter-session data)
- [ ] Re-calibration trigger when SNR drops > 3 dB vs baseline (software hook exists)

---

## What not to build yet

| Temptation | Why defer |
|------------|-----------|
| 22+ channel HD array | Gowda 2024 research setup; 8–10 ch sufficient ([Ji 2021](https://doi.org/10.1088/1741-2552/abca14)) |
| 5 kHz sampling | Research only; 250–1000 Hz adequate for command vocab |
| Custom ASIC / flexible printed electronics | Wang 2021 scale; use COTS for V0–V2 |
| Clinical certification (IEC 60601) | Research prototype; follow safety practices, not full QMS |

---

## Software-first parallel path

While hardware tiers progress, use:

```bash
openalterego sim-dataset --out ./session
openalterego serve --source virtual_ble   # packet loss / jitter
```

Virtual BLE exercises the same OA v1 path as real firmware ([`acquisition/virtual.py`](../software/python/openalterego/acquisition/virtual.py)).
