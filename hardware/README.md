# OpenAlterEgo Hardware Design

Systematic hardware documentation for the silent-speech EMG acquisition path — from electrodes to BLE packets.

**Software counterpart:** `software/python/openalterego/acquisition/` (OA v1 packet format, BLE client, virtual link)

**ML/DSP counterpart:** `docs/12-references.md`, `docs/literature/README.md`, `docs/14-systematic-roadmap.md`

---

## Design philosophy

1. **Tiered progression** — benchtop (V0) → wearable PCB (V1) → mechanical repeatability (V2)
2. **Literature-first parameters** — every major spec traces to a paper or datasheet
3. **Simulation before silicon** — validate DSP/ML on sim + virtual BLE before custom firmware
4. **Safety by default** — battery-powered, isolated debug, minimal patient leakage current

---

## Document map

| Doc | Topic |
|-----|-------|
| [block_diagram.md](block_diagram.md) | End-to-end signal path (electrodes → host) |
| [01-architecture-tiers.md](01-architecture-tiers.md) | V0 / V1 / V2 milestones and exit criteria |
| [02-analog-front-end.md](02-analog-front-end.md) | AFE selection, gain, sampling rate, noise budget |
| [03-electrodes-montage.md](03-electrodes-montage.md) | Muscle targets, channel count, dry vs wet, references |
| [04-ble-firmware-protocol.md](04-ble-firmware-protocol.md) | MCU, BLE stack, OA v1 packet framing |
| [05-power-safety.md](05-power-safety.md) | Power tree, isolation, ESD, regulatory checklist |
| [06-mechanical-wearable.md](06-mechanical-wearable.md) | Form factors, cable routing, repeatability |
| [BOM.md](BOM.md) | Bill of materials by tier with part rationale |
| [07-references.md](07-references.md) | Hardware-specific bibliography |
| [08-hardware-dsl.md](08-hardware-dsl.md) | Runnable `.oae.json` DSL |
| [09-simulation-realism.md](09-simulation-realism.md) | Sim-to-real ladder (biophysical, tang SNR, montage) |
| [10-neurobiophysical-emg.md](10-neurobiophysical-emg.md) | Neurobiophysical EMG architecture (v5 motor pool) |
| [specs/](specs/) | Example spec files |

---

## Quick parameter summary (literature-aligned)

| Parameter | OpenAlterEgo target | Primary literature |
|-----------|---------------------|-------------------|
| Channels | **7–8** differential (4 minimum viable) | Kapur 2018; Wang 2021 (4 ch); Tang 2025 (4 ch) |
| ADC | **24-bit** class | Kapur 2020; ADS1299 family |
| Sample rate | **250 Hz** default; **500–1000 Hz** optional | Kapur 2018 (250); Tang 2025 (1000); Wang 2021 (500) |
| Gain | **24×** (AlterEgo); **6–12×** for high-fs wideband | Kapur 2018; SilentWear (PGA 6 @ 500 Hz) |
| Bandwidth (AFE) | **0.5–500 Hz** analog headroom; DSP sets effective band | Wide EMG papers use 20–450 Hz DSP |
| Reference | Earlobe or wrist; BIAS/DRL active | Kapur 2018/2020; ADS1299 app notes |
| Transport | **BLE** notifications | Kapur 2018 patent; OA v1 framing in repo |
| Electrodes (V0) | Pre-gelled **Ag/AgCl** | Kapur 2018 (Ten20 paste); EMG-UKA |
| Electrodes (V1+) | Dry textile or conductive polymer | Tang 2025; SilentWear 2025; Wang 2021 tattoo |

---

## Hardware DSL (simulate before you solder)

Use **`.oae.json`** specs to validate, resolve, and simulate acquisition stacks:

```bash
cd software/python
uv run openalterego hw validate v0_openbci
uv run openalterego hw simulate v1_wearable_ble --path both --seconds 3
```

See [**08-hardware-dsl.md**](08-hardware-dsl.md) and [`specs/`](specs/).

---

```
V0 benchtop (OpenBCI / ADS1299 dev kit)
    → validate SNR + placement + 250 Hz pipeline
V1 wearable PCB (ADS1299 + nRF52840 + LiPo)
    → OA v1 firmware + BLE field test
V2 mechanical frame (AlterEgo-style head band OR neckband OR headphone)
    → repeatable montage + multi-day wear
```

See [01-architecture-tiers.md](01-architecture-tiers.md) for exit criteria per tier.
