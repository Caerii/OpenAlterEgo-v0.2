# Hardware

This project intentionally supports *multiple* hardware tiers:

1. **V0 Benchtop rig (fast):** Off-the-shelf biopotential AFE dev board + adhesive electrodes.
2. **V1 Wearable prototype:** Custom PCB with low-noise AFE + BLE microcontroller + battery.
3. **V2 Comfort + repeatability:** Mechanical design to keep electrodes stable, repeatable, adjustable.

---

## V0: Benchtop rig (recommended first)

### Why
Because the hard parts are:
- getting clean signals, and
- getting repeatable electrode placement.

A benchtop rig lets you debug DSP + ML without fighting mechanical design.

### Suggested parts (examples)
- 8-channel biopotential AFE board (ADS1299-based) OR an OpenBCI Cyton/Ganglion.
- Pre-gelled Ag/AgCl electrodes (or dry electrodes if you want “wearable” pain now).
- Shielded electrode leads.
- Battery power for the AFE and any laptop connection via **USB isolator** if you’re not 100% sure.

---

## V1: Wearable PCB architecture (block level)

### Must-haves
- 7–8 differential channels
- 24-bit ADC class
- Input protection (ESD + series resistors)
- Bias/DRL-style circuit if your AFE supports it
- BLE radio (nRF52-class tends to be quiet and power efficient)
- Battery + charger (LiPo) + proper power tree (analog clean rail)

### Nice-to-haves
- On-board IMU (useful to model head motion artifacts)
- Electrode impedance measurement (for quality checks)
- Trigger button or marker channel

---

## Electrode placement (research-friendly, not medical advice)

**General principle:** you want a mix of face/jaw and neck regions that correlate with articulation.

Public work reports channels roughly in:
- chin / mentum region
- around the lips / mouth corners (orbicularis oris / levator anguli oris region)
- under the jaw (digastric area)
- front neck (hyoid / laryngeal / platysma regions)

**Reference & bias:** later work describes using reference + bias electrodes on the earlobes.

> Your placement will be user-specific. Expect to tune it.

---

## Electrical safety checklist (do not skip)

- ✅ Battery-powered while worn.
- ✅ No direct galvanic path from the user to mains earth.
- ✅ If you debug while tethered to a laptop: use a **medical-grade** or at least high-quality USB isolator.
- ✅ Keep leakage current tiny: series resistors + ESD protection.
- ✅ Enclosure: strain relief so leads can’t yank electrodes and tear skin.

---

## Mechanical design notes

Repeatability beats cleverness.

- Make electrode arms **rigid** but adjustable.
- Use spring compliance only where it improves comfort without introducing slop.
- Minimize cable motion: route wires along the frame, not dangling.
- If you’re using gel electrodes, design for easy re-application and cleaning.



---

## Simulation-first workflow

Before you build hardware, this repo lets you iterate end-to-end in software:

- synthetic multichannel signals (`openalterego.sim`)
- “virtual BLE” notifications with packet loss/jitter (`openalterego.acquisition.virtual`)
- OpenAlterEgo v1 packet framing (`openalterego.acquisition.packet`)
- realtime websocket output (`openalterego.api.server`)

If/when you implement firmware, prefer the **OpenAlterEgo v1** framed packet format so the host can detect packet loss
and survive BLE MTU changes.
