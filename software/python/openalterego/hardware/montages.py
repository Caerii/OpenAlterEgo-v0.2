"""Electrode montage presets (literature-aligned)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class MontagePreset:
    name: str
    channels: int
    sites: List[str]
    literature_ref: str
    notes: str = ""


MONTAGES: Dict[str, MontagePreset] = {
    "alterego_8ch": MontagePreset(
        name="alterego_8ch",
        channels=8,
        sites=[
            "laryngeal",
            "hyoid",
            "levator_anguli_oris",
            "orbicularis_oris",
            "platysma",
            "digastric",
            "mentum",
            "masseter_or_scm",
        ],
        literature_ref="Kapur et al. 2018 IUI",
        notes="Default OpenAlterEgo face/neck layout",
    ),
    "kapur_2020_clinical": MontagePreset(
        name="kapur_2020_clinical",
        channels=8,
        sites=[
            "face_1",
            "face_2",
            "face_3",
            "face_4",
            "neck_1",
            "neck_2",
            "neck_3",
            "neck_4",
        ],
        literature_ref="Kapur et al. 2020 ML4H",
        notes="4 face + 4 neck; earlobe ref/bias",
    ),
    "wang_4ch": MontagePreset(
        name="wang_4ch",
        channels=4,
        sites=["LAO", "DAO", "BUC", "ABD"],
        literature_ref="Wang et al. 2021 npj Flexible Electronics",
    ),
    "tang_4ch_headphone": MontagePreset(
        name="tang_4ch_headphone",
        channels=4,
        sites=["headphone_emg_1", "headphone_emg_2", "headphone_emg_3", "headphone_emg_4"],
        literature_ref="Tang et al. 2025 IEEE TIM",
        notes="Textile electrodes in earmuff",
    ),
    "lai_3ch": MontagePreset(
        name="lai_3ch",
        channels=3,
        sites=["LAO", "DAO", "ZM"],
        literature_ref="Lai et al. 2023 arXiv:2308.06533",
    ),
    "silentwear_10ch": MontagePreset(
        name="silentwear_10ch",
        channels=10,
        sites=[f"neck_diff_{i}" for i in range(1, 11)],
        literature_ref="Meier et al. 2025 SilentWear arXiv:2603.02847",
        notes="Overlapping differential pairs on neckband",
    ),
    "deng_8ch": MontagePreset(
        name="deng_8ch",
        channels=8,
        sites=["ZYG", "RIS", "DAO", "SCM", "ABD", "PLT", "site_7", "site_8"],
        literature_ref="Deng et al. 2023 IEEE TIM",
    ),
    "gowda_31ch": MontagePreset(
        name="gowda_31ch",
        channels=31,
        sites=[f"gowda_site_{i}" for i in range(1, 32)],
        literature_ref="Gowda et al. 2025 arXiv:2502.05762",
        notes="OSF small/large vocab lab array (31 ch @ 5 kHz); 1D arc proxy for forward pickup",
    ),
}


def get_montage(name: str) -> MontagePreset:
    key = str(name).strip()
    if key not in MONTAGES:
        known = ", ".join(sorted(MONTAGES))
        raise ValueError(f"unknown montage {name!r}; known: {known}")
    return MONTAGES[key]


def list_montage_names() -> List[str]:
    return sorted(MONTAGES.keys())
