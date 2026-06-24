# Bill of materials (BOM)

Tiered parts list with literature rationale. **Not a finalized production BOM** — a decision menu for prototypes.

---

## V0 Benchtop

**Goal:** 8-ch, 250 Hz, gel electrodes, fastest path to real EMG in `openalterego calibrate`.

### Option A — OpenBCI Cyton (recommended)

| Qty | Part | Est. cost | Notes |
|-----|------|-----------|-------|
| 1 | [OpenBCI Cyton](https://shop.openbci.com/collections/frontpage/products/cyton-biosensing-board-8-channel) | ~$500 | 8× ADS1299, USB + BLE dongle; AlterEgo 2018 reference platform |
| 1 | OpenBCI Ultracortex Mark IV (or DIY cap) | ~$350 | Mechanical holder; optional for V0 |
| 10 | Ag/AgCl pre-gelled electrodes + leads | ~$30 | Disposable; Kapur 2018 Ten20-compatible |
| 1 | Ten20 conductive paste (optional) | ~$20 | Kapur 2018 preferred paste |
| 1 | USB isolator (medical-grade preferred) | ~$80–200 | [05-power-safety.md](05-power-safety.md) |
| 1 | USB battery pack | ~$25 | Field laptop power |

**Literature fit:** 24-bit, 8 ch, community support, matches Kapur 2018 stack.

### Option B — TI ADS1299 FE + USB controller

| Qty | Part | Notes |
|-----|------|-------|
| 1 | ADS1299EEG-FE PD kit | TI official FE; SPI to PC via adapter |
| 8 | Snap electrode cables | |
| 10 | Ag/AgCl electrodes | |

**Literature fit:** Closest to custom V1 analog path; more firmware work on V0.

### Option C — OpenBCI Ganglion (budget 4-ch)

| Qty | Part | Notes |
|-----|------|-------|
| 1 | Ganglion board | 4 ch — matches Wang 2021 / Tang 2025 minimum |

---

## V1 Wearable PCB

**Goal:** Battery-powered BLE stream in OA v1 format.

### Analog / acquisition

| Qty | Part | Mfr | Notes |
|-----|------|-----|-------|
| 1 | ADS1299-8 IPAGR | TI | 8-ch 24-bit; [02-analog-front-end.md](02-analog-front-end.md) |
| 1 | ADS1299EEG-PDK (ref design) | TI | Layout reference |
| 8 | 100kΩ 0603 series input R | — | Patient protection |
| 8 | ESD protection (e.g. TPD1E10B06) | TI | Per input |
| 2–4 | 1 µF / 0.1 µF AVDD decoupling | — | Per datasheet layout |

### Digital / radio

| Qty | Part | Notes |
|-----|------|-------|
| 1 | nRF52840 (module e.g. Raytac MDBT50) | BLE 5.0; [04-ble-firmware-protocol.md](04-ble-firmware-protocol.md) |
| 1 | SPI level shifter (if needed) | 3.3 V typical both sides |

### Power

| Qty | Part | Notes |
|-----|------|-------|
| 1 | MCP73831 LiPo charger | USB-C input |
| 1 | LiPo 3.7 V 300–500 mAh | > 4 h target |
| 2 | Low-noise LDO (e.g. TPS7A20 + AP2112) | Analog + digital rails |
| 1 | Load switch (e.g. TPS22965) | Soft power on |

### Connectors / mechanical

| Qty | Part | Notes |
|-----|------|-------|
| 8–10 | Snap electrode connectors (touch-proof) | DIN 42802 class preferred |
| 1 | JST PH housing for electrode harness | Strain relief |
| 1 | Enclosure 80×50×15 mm approx | Head/neck rear mount |

### Optional

| Qty | Part | Literature |
|-----|------|------------|
| 1 | BMI270 IMU | Motion artifact context |
| 1 | MAX17048 fuel gauge | Battery SOC |

**Reference system:** BioGAP-Ultra (2× ADS1298, nRF5340, 22 mW) — [SilentWear](https://arxiv.org/abs/2603.02847)

---

## V2 Mechanical / electrodes

### AlterEgo-style head band

| Qty | Part | Notes |
|-----|------|-------|
| 1 | PETG/ABS printed band + 7 arms | [06-mechanical-wearable.md](06-mechanical-wearable.md) |
| 7 | Brass rod or CF tube segments | AlterEgo 2018 |
| 7 | Gold-plated or Ag/AgCl cup electrodes | Kapur 2018 |
| 1 | Velcro / elastic rear strap | Head size adjustment |

### Neckband (SilentWear-style)

| Qty | Part | Notes |
|-----|------|-------|
| 1 | Fabric neckband blank | Commercial or sewn |
| 15–27 | Snap fasteners (sewn) | SilentWear: 27 positions |
| 15 | Datwyler SoftPulse or equiv. dry electrode | [arXiv:2603.02847](https://arxiv.org/abs/2603.02847) |
| 1 | Velcro sizing strap | |

### Headphone integration (Tang-style)

| Qty | Part | Notes |
|-----|------|-------|
| 1 | Over-ear headphone shell | Donor pair |
| 4 | Graphene/PEDOT:PSS textile electrodes | Custom fab — research |
| 1 | ESP32-S3-MINI module | Tang 2025 readout |

---

## Output / feedback (optional)

| Qty | Part | Notes |
|-----|------|-------|
| 1 | Bone-conduction transducer (e.g. AfterShokz donor) | AlterEgo patent audio feedback |
| 1 | Low-profile earbud | Dev monitoring |

---

## Tools and test equipment

| Item | Purpose |
|------|---------|
| Oscilloscope + differential probe | AFE bring-up |
| Function generator + attenuator | Gain calibration |
| BLE sniffer (nRF Sniffer) | Packet timing |
| Impedance analyzer (optional) | Dry electrode development |

---

## Cost targets (rough)

| Tier | BOM (excl. tools) | Notes |
|------|-------------------|-------|
| V0 OpenBCI | ~$900 | Fastest |
| V1 custom PCB (10 units) | ~$150–250/unit | PCBA + passives |
| V2 mechanical | +$50–200 | Prints, electrodes |

---

## Software mapping

| Hardware output | Software entry |
|-----------------|----------------|
| OA v1 BLE notify | `acquisition/ble_client.py` → `collect ble` |
| OpenBCI raw stream | Adapter or firmware shim to OA v1 |
| Sim / virtual | `acquisition/virtual.py` — no hardware |

---

## Related

- Tier exit criteria: [01-architecture-tiers.md](01-architecture-tiers.md)
- Hardware bibliography: [07-references.md](07-references.md)
