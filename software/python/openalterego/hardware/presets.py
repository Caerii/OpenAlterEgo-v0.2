"""Built-in hardware spec presets (literature-aligned)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

from .schema import HardwareSpec

# Each preset is a JSON-serializable dict validated by HardwareSpec.
PRESETS: Dict[str, Dict[str, Any]] = {
    "v0_openbci": {
        "schema_version": 1,
        "name": "v0_openbci",
        "tier": "v0",
        "description": "V0 benchtop: OpenBCI Cyton-class 8ch @ 250 Hz, wet gel, AlterEgo montage",
        "literature_refs": ["Kapur 2018 IUI", "TI ADS1299", "OpenBCI Cyton"],
        "afe": {
            "part": "ads1299",
            "channels": 8,
            "fs_hz": 250,
            "gain": 24.0,
            "adc_bits": 16,
            "internal_adc_bits": 24,
            "vref_v": 2.4,
        },
        "electrodes": {
            "type": "wet_ag_agcl",
            "montage": "alterego_8ch",
            "reference": "earlobe",
        },
        "preprocess": {"mode": "standard", "notch_hz": 60.0, "notch_harmonics": True},
        "ble": {
            "transport": "usb",
            "packet_format": "oa_v1",
            "frames_per_packet": 12,
            "device_name": "OpenBCICyton",
        },
        "link": {"loss_prob": 0.0, "jitter_ms": 0.0, "extra_latency_ms": 0.0, "seed": 123},
        "sim": {
            "engine": "biophysical",
            "realism": "wearable",
            "emg_paradigm": "alterego_envelope",
            "seed": 1337,
            "noise_uV": 18.0,
            "line_noise_uV": 8.0,
            "snr_target_static_db": 20.0,
        },
    },
    "alterego_2018": {
        "schema_version": 1,
        "name": "alterego_2018",
        "tier": "v2",
        "description": "AlterEgo 2018 reported parameters: 7-8ch, 250 Hz, 24x gain, standard envelope band",
        "literature_refs": ["Kapur 2018 IUI", "US10878818B2"],
        "afe": {
            "part": "ads1299",
            "channels": 8,
            "fs_hz": 250,
            "gain": 24.0,
            "adc_bits": 16,
            "internal_adc_bits": 24,
            "vref_v": 2.4,
        },
        "electrodes": {
            "type": "dry_metal",
            "montage": "alterego_8ch",
            "reference": "wrist",
        },
        "preprocess": {"mode": "standard", "notch_hz": 60.0, "notch_harmonics": True},
        "ble": {"transport": "ble", "packet_format": "oa_v1", "frames_per_packet": 12},
        "sim": {
            "engine": "biophysical",
            "realism": "wearable",
            "emg_paradigm": "alterego_envelope",
        },
    },
    "kapur_2020_clinical": {
        "schema_version": 1,
        "name": "kapur_2020_clinical",
        "tier": "v0",
        "description": "Clinical MS dysphonia layout: 8ch, 250 Hz, clinical 0.5-8 Hz DSP",
        "literature_refs": ["Kapur 2020 ML4H"],
        "afe": {"channels": 8, "fs_hz": 250, "gain": 24.0},
        "electrodes": {
            "type": "wet_ag_agcl",
            "montage": "kapur_2020_clinical",
            "reference": "earlobe",
        },
        "preprocess": {"mode": "clinical", "notch_hz": 60.0, "notch_harmonics": True},
        "sim": {"engine": "biophysical", "realism": "wearable", "emg_paradigm": "alterego_envelope"},
    },
    "v1_wearable_ble": {
        "schema_version": 1,
        "name": "v1_wearable_ble",
        "tier": "v1",
        "description": "V1 custom PCB: ADS1299 + nRF52840 BLE, wide DSP @ 500 Hz",
        "literature_refs": ["Wang 2021", "Tang 2025", "TI ADS1299"],
        "afe": {
            "part": "ads1299",
            "channels": 8,
            "fs_hz": 500,
            "gain": 12.0,
            "adc_bits": 16,
            "internal_adc_bits": 24,
            "vref_v": 2.4,
        },
        "electrodes": {
            "type": "dry_textile",
            "montage": "alterego_8ch",
            "reference": "earlobe",
        },
        "preprocess": {"mode": "wide", "notch_hz": 60.0, "notch_harmonics": True},
        "ble": {
            "transport": "ble",
            "packet_format": "oa_v1",
            "frames_per_packet": 12,
            "device_name": "OpenAlterEgo",
        },
        "link": {"loss_prob": 0.01, "jitter_ms": 2.0, "extra_latency_ms": 5.0},
        "sim": {
            "engine": "biophysical",
            "realism": "tang",
            "emg_paradigm": "semg_literature_clamped",
            "noise_uV": 28.0,
            "line_noise_uV": 12.0,
            "snr_target_static_db": 18.9,
        },
    },
    "tang_2025_headphone": {
        "schema_version": 1,
        "name": "tang_2025_headphone",
        "tier": "v2",
        "description": "Headphone textile EMG: 4ch @ 1000 Hz, wide band, dry textile",
        "literature_refs": ["Tang et al. 2025 arXiv:2504.13921"],
        "afe": {
            "channels": 4,
            "fs_hz": 1000,
            "gain": 12.0,
            "adc_bits": 16,
            "vref_v": 2.4,
        },
        "electrodes": {
            "type": "dry_textile",
            "montage": "tang_4ch_headphone",
            "reference": "differential_only",
        },
        "preprocess": {"mode": "wide", "notch_hz": 60.0, "notch_harmonics": True},
        "ble": {"transport": "ble", "packet_format": "oa_v1", "device_name": "EMG-Headphone"},
        "sim": {
            "engine": "biophysical",
            "realism": "tang",
            "emg_paradigm": "semg_literature_clamped",
            "noise_uV": 32.0,
            "line_noise_uV": 15.0,
            "snr_target_static_db": 18.9,
        },
    },
    "silentwear_2025": {
        "schema_version": 1,
        "name": "silentwear_2025",
        "tier": "v2",
        "description": "SilentWear neckband: 10 diff ch @ 500 Hz, dry textile, PGA gain 6",
        "literature_refs": ["Meier et al. 2025 arXiv:2603.02847", "BioGAP-Ultra"],
        "afe": {
            "part": "ads1298",
            "channels": 10,
            "fs_hz": 500,
            "gain": 6.0,
            "adc_bits": 16,
            "vref_v": 2.4,
        },
        "electrodes": {
            "type": "dry_textile",
            "montage": "silentwear_10ch",
            "reference": "neck_ground",
        },
        "preprocess": {"mode": "wide", "notch_hz": 50.0, "notch_harmonics": True},
        "ble": {"transport": "ble", "packet_format": "oa_v1", "device_name": "SilentWear"},
        "sim": {
            "engine": "biophysical",
            "realism": "wearable",
            "emg_paradigm": "semg_literature_clamped",
            "noise_uV": 24.0,
            "snr_target_static_db": 18.9,
        },
    },
    "wang_2021_tattoo": {
        "schema_version": 1,
        "name": "wang_2021_tattoo",
        "tier": "v2",
        "description": "Tattoo-like flexible 4ch @ 500 Hz",
        "literature_refs": ["Wang et al. 2021 npj Flexible Electronics"],
        "afe": {"channels": 4, "fs_hz": 500, "gain": 24.0},
        "electrodes": {
            "type": "tattoo_flex",
            "montage": "wang_4ch",
            "reference": "earlobe",
        },
        "preprocess": {"mode": "wide", "notch_hz": 60.0, "notch_harmonics": True},
        "sim": {
            "engine": "biophysical",
            "realism": "tang",
            "emg_paradigm": "semg_literature_clamped",
            "snr_target_static_db": 18.9,
        },
    },
    "gowda_31ch_5khz": {
        "schema_version": 1,
        "name": "gowda_31ch_5khz",
        "tier": "v2",
        "description": "Gowda OSF small/large vocab: 31 ch @ 5 kHz, wide DSP, biophysical sim aligned to CTC benchmark",
        "literature_refs": ["Gowda et al. 2025 arXiv:2502.05762", "emg2speech OSF"],
        "afe": {
            "channels": 31,
            "fs_hz": 5000,
            "gain": 24.0,
            "adc_bits": 16,
            "internal_adc_bits": 24,
            "vref_v": 2.4,
        },
        "electrodes": {
            "type": "wet_ag_agcl",
            "montage": "gowda_31ch",
            "reference": "earlobe",
        },
        "preprocess": {"mode": "wide", "notch_hz": 60.0, "notch_harmonics": True},
        "sim": {
            "engine": "biophysical",
            "realism": "tang",
            "emg_paradigm": "semg_literature_clamped",
            "noise_uV": 28.0,
            "line_noise_uV": 12.0,
            "snr_target_static_db": 18.9,
            "snr_motion_target_db": 12.7,
        },
    },
}


def preset_names() -> list[str]:
    return sorted(PRESETS.keys())


def get_preset_dict(name: str) -> Dict[str, Any]:
    key = str(name).strip()
    if key not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; known: {', '.join(preset_names())}")
    return deepcopy(PRESETS[key])


def load_preset(name: str) -> HardwareSpec:
    return HardwareSpec.model_validate(get_preset_dict(name))
