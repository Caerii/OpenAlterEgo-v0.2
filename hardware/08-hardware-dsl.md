# Hardware DSL (`.oae.json`)

Runnable, simulatable hardware specifications that bind literature-aligned acquisition parameters to the Python stack.

**Implementation:** `software/python/openalterego/hardware/`

---

## Quick start

```bash
cd software/python

# List built-in presets and montages
uv run openalterego hw list

# Validate literature constraints
uv run openalterego hw validate v0_openbci
uv run openalterego hw validate ../../hardware/specs/v1_wearable_ble.oae.json

# Resolve → sim + virtual BLE bindings
uv run openalterego hw resolve v1_wearable_ble --json

# Run short simulation (direct chunks + OA v1 byte path)
uv run openalterego hw simulate v0_openbci --seconds 3
uv run openalterego hw simulate v1_wearable_ble --path virtual_ble --seconds 2
```

---

## File format

Extension: **`.oae.json`** (Open Alter Ego hardware spec). Schema version **1**.

```json
{
  "schema_version": 1,
  "name": "my_lab_rig",
  "tier": "v0",
  "description": "Benchtop rig",
  "literature_refs": ["Kapur 2018 IUI"],
  "extends": "v0_openbci",
  "afe": {
    "channels": 8,
    "fs_hz": 250,
    "gain": 24.0,
    "adc_bits": 16,
    "vref_v": 2.4
  },
  "electrodes": {
    "type": "wet_ag_agcl",
    "montage": "alterego_8ch",
    "reference": "earlobe"
  },
  "preprocess": { "mode": "standard", "notch_hz": 60.0, "notch_harmonics": true },
  "ble": {
    "transport": "ble",
    "packet_format": "oa_v1",
    "frames_per_packet": 12,
    "device_name": "OpenAlterEgo"
  },
  "link": { "loss_prob": 0.0, "jitter_ms": 0.0 },
  "sim": {
    "engine": "heuristic",
    "realism": "wearable",
    "noise_uV": 22.0
  }
}
```

### `extends`

Child specs shallow-merge over a **preset name** or another `.oae.json` file:

```json
{
  "name": "custom_wide_lab",
  "extends": "v1_wearable_ble",
  "sim": { "noise_uV": 40.0 }
}
```

Example: [`specs/custom_wide_lab.oae.json`](specs/custom_wide_lab.oae.json)

---

## Built-in presets

| Preset | Tier | fs | Channels | Preprocess | Literature |
|--------|------|-----|----------|------------|------------|
| `v0_openbci` | v0 | 250 | 8 | standard | Kapur 2018, OpenBCI |
| `alterego_2018` | v2 | 250 | 8 | standard | Kapur 2018 |
| `kapur_2020_clinical` | v0 | 250 | 8 | clinical | Kapur 2020 |
| `v1_wearable_ble` | v1 | 500 | 8 | wide | Wang/Tang |
| `tang_2025_headphone` | v2 | 1000 | 4 | wide | Tang 2025 |
| `silentwear_2025` | v2 | 500 | 10 | wide | SilentWear |
| `wang_2021_tattoo` | v2 | 500 | 4 | wide | Wang 2021 |

---

## Montage presets

| Name | Channels | Literature |
|------|----------|------------|
| `alterego_8ch` | 8 | Kapur 2018 |
| `kapur_2020_clinical` | 8 | Kapur 2020 |
| `wang_4ch` | 4 | Wang 2021 |
| `tang_4ch_headphone` | 4 | Tang 2025 |
| `lai_3ch` | 3 | Lai 2023 |
| `silentwear_10ch` | 10 | SilentWear 2025 |
| `deng_8ch` | 8 | Deng 2023 |

Defined in `openalterego/hardware/montages.py`.

---

## What `validate` checks

| Code | Severity | Meaning |
|------|----------|---------|
| `preprocess.wide_fs_low` | warning | Wide DSP at fs &lt; 500 Hz (literature uses 500–1000 Hz) |
| `afe.gain_dry` | warning | High gain with dry electrodes |
| `ble.mtu` | error | OA v1 packet exceeds BLE MTU budget |
| `sim.paradigm_fs` | error | `emg_paradigm` incompatible with `fs_hz` |
| `channels.montage_mismatch` | warning | `afe.channels` ≠ montage default |

---

## Resolution pipeline

```
.oae.json / preset
    → validate_spec()
    → resolve_all()
        → SimConfig          → stream_simulated_chunks()
        → VirtualBleSpec       → virtual_notifications() + OA v1 parse
        → AfeSpec              → µV scaling on wire
        → preprocess_mode      → dsp/emg_config.py
        → emg_paradigm         → sim/literature.py
```

---

## End-to-end pipeline (`--hw-spec`)

Hardware specs bind acquisition parameters through collect, dataset generation, and serve:

```bash
cd software/python

# One-shot: validate → smoke sim → collect session
uv run openalterego hw run v0_openbci --out ./session --user-id alice --seconds 60

# Collect with preset (writes hardware_spec into session.json)
uv run openalterego collect sim --hw-spec v0_openbci --out ./session --user-id alice --seconds 60

# Generate synthetic dataset aligned to spec
uv run openalterego sim-dataset --hw-spec v1_wearable_ble --out ./dataset --seconds 120

# Serve with sim source bound to spec (fs, channels, preprocess, packet format)
uv run openalterego serve --source sim --hw-spec v0_openbci --user-id alice
```

`--hw-spec` accepts a **preset name** (`v0_openbci`, `v1_wearable_ble`, …) or a path to `.oae.json`. When set, CLI flags for fs/channels/preprocess are taken from the spec unless you use the legacy collect path without `--hw-spec`.

Typical workflow after `hw run`:

```bash
openalterego calibrate --user-id alice --data ./session --fs 250
openalterego train --user-id alice --data ./session --fs 250 --emg-mode standard
openalterego serve --source sim --user-id alice --hw-spec v0_openbci
```

---

## Repo spec files

| File | Purpose |
|------|---------|
| [`specs/v0_openbci.oae.json`](specs/v0_openbci.oae.json) | Default benchtop |
| [`specs/v1_wearable_ble.oae.json`](specs/v1_wearable_ble.oae.json) | V1 wearable target |
| [`specs/custom_wide_lab.oae.json`](specs/custom_wide_lab.oae.json) | `extends` example |

---

## CLI reference

```
openalterego hw list [--json]
openalterego hw show <spec> [--json]
openalterego hw validate <spec>
openalterego hw resolve <spec> [--json]
openalterego hw simulate <spec> [--seconds N] [--path chunks|virtual_ble|both] [--json]
openalterego hw run <spec> --out DIR [--user-id ID] [--seconds N] [--smoke-seconds N] [--skip-smoke] [--json]
openalterego hw export <preset> -o path.oae.json
```

Pipeline commands accepting `--hw-spec`:

```
openalterego collect sim --hw-spec <spec> ...
openalterego sim-dataset --hw-spec <spec> ...
openalterego serve --source sim|virtual_ble|ble --hw-spec <spec> ...
```

`<spec>` = preset name **or** path to `.oae.json`.

---

## Related

- Block diagram: [block_diagram.md](block_diagram.md)
- OA v1 packet: `software/python/openalterego/acquisition/packet.py`
- Virtual BLE: `software/python/openalterego/acquisition/virtual.py`
- Tests: `software/python/tests/test_hardware_dsl.py`
