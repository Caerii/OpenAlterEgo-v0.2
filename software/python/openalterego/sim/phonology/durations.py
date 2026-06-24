"""ARPABET phone duration priors for within-word pseudo-alignment."""

from __future__ import annotations

from typing import Dict, List, Sequence

# Relative duration weights (unitless); vowels and diphthongs longer, stops shorter.
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "AA": 1.0, "AE": 0.95, "AH": 0.85, "AO": 1.0, "AW": 1.1, "AY": 1.1,
    "B": 0.35, "CH": 0.45, "D": 0.35, "DH": 0.4, "EH": 0.9, "ER": 0.95, "EY": 1.0,
    "F": 0.5, "G": 0.4, "HH": 0.35, "IH": 0.85, "IY": 0.9, "JH": 0.45, "K": 0.35,
    "L": 0.55, "M": 0.5, "N": 0.5, "NG": 0.55, "OW": 1.0, "OY": 1.05, "P": 0.35,
    "R": 0.55, "S": 0.5, "SH": 0.55, "T": 0.35, "TH": 0.45, "UH": 0.85, "UW": 0.95,
    "V": 0.5, "W": 0.55, "Y": 0.55, "Z": 0.5, "ZH": 0.55,
}
_FALLBACK_WEIGHT = 0.6


def phone_duration_weight(phone: str) -> float:
    key = str(phone).strip().upper()
    return float(_DEFAULT_WEIGHTS.get(key, _FALLBACK_WEIGHT))


def phone_duration_weights(phones: Sequence[str]) -> List[float]:
    return [phone_duration_weight(p) for p in phones]
