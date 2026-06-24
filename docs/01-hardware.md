# Hardware

> **Canonical hardware documentation** lives in [`hardware/`](../hardware/README.md). This page is a short index.

---

## Start here

| Document | Content |
|----------|---------|
| [**hardware/README.md**](../hardware/README.md) | Overview, parameter summary, build order |
| [**block_diagram.md**](../hardware/block_diagram.md) | Electrodes → AFE → BLE → host |
| [**01-architecture-tiers.md**](../hardware/01-architecture-tiers.md) | V0 benchtop → V1 PCB → V2 mechanical |
| [**02-analog-front-end.md**](../hardware/02-analog-front-end.md) | ADS1299, gain, fs, noise budget |
| [**03-electrodes-montage.md**](../hardware/03-electrodes-montage.md) | Muscle map, dry vs wet, 8-ch layout |
| [**04-ble-firmware-protocol.md**](../hardware/04-ble-firmware-protocol.md) | OA v1 packet, nRF52, firmware workflow |
| [**05-power-safety.md**](../hardware/05-power-safety.md) | LiPo tree, isolation, safety checklist |
| [**06-mechanical-wearable.md**](../hardware/06-mechanical-wearable.md) | Head band, neckband, cable routing |
| [**BOM.md**](../hardware/BOM.md) | Parts list by tier |
| [**07-references.md**](../hardware/07-references.md) | Hardware-specific bibliography |

---

## Three tiers (summary)

1. **V0 Benchtop** — OpenBCI / ADS1299 dev kit + gel electrodes. Debug DSP/ML without custom PCB.
2. **V1 Wearable PCB** — ADS1299 + nRF52840 + LiPo + OA v1 BLE firmware.
3. **V2 Comfort** — Mechanical frame (AlterEgo head band, neckband, or headphone) for repeatable placement.

Exit criteria per tier: [01-architecture-tiers.md](../hardware/01-architecture-tiers.md).

---

## Literature-aligned defaults

| Parameter | Default | Primary reference |
|-----------|---------|-------------------|
| Channels | 7–8 (4 min) | Kapur 2018; Wang 2021 |
| Sample rate | 250 Hz (500–1000 for wide DSP) | Kapur 2018; Tang 2025 |
| ADC / gain | 24-bit, PGA 24× | Kapur 2018/2020; ADS1299 |
| Reference | Earlobe BIAS | Kapur 2020 |
| Transport | BLE + OA v1 packets | Kapur 2018; `acquisition/packet.py` |

Full bibliography: [hardware/07-references.md](../hardware/07-references.md) and [12-references.md](12-references.md).

---

## Software integration

Before building custom firmware:

```bash
openalterego sim-dataset --out ./session
openalterego serve --source virtual_ble
```

Firmware should emit **OpenAlterEgo v1** packets (`software/python/openalterego/acquisition/packet.py`) for drop-in host compatibility.

---

## Simulation-first workflow

The repo supports full end-to-end iteration without hardware:

- Synthetic multichannel EMG (`openalterego.sim`)
- Virtual BLE with loss/jitter (`openalterego.acquisition.virtual`)
- Realtime WebSocket tokens (`openalterego.api.server`)

See [14-systematic-roadmap.md](14-systematic-roadmap.md) Phase B for hardware bring-up alongside Phase A validation.
