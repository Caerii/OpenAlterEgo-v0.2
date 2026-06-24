"""Load and apply per-phone templates (channel profiles, rate scales) for biophysical drive."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


@dataclass
class PhoneTemplate:
    phone: str
    channel_rms: np.ndarray
    spd_diag_delta: np.ndarray
    rate_scale: float
    duration_weight: float
    n_segments: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "phone": str(self.phone),
            "channel_rms": [float(x) for x in self.channel_rms.tolist()],
            "spd_diag_delta": [float(x) for x in self.spd_diag_delta.tolist()],
            "rate_scale": float(self.rate_scale),
            "duration_weight": float(self.duration_weight),
            "n_segments": int(self.n_segments),
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "PhoneTemplate":
        return cls(
            phone=str(row["phone"]),
            channel_rms=np.asarray(row["channel_rms"], dtype=np.float64),
            spd_diag_delta=np.asarray(row["spd_diag_delta"], dtype=np.float64),
            rate_scale=float(row.get("rate_scale", 1.0)),
            duration_weight=float(row.get("duration_weight", 0.6)),
            n_segments=int(row.get("n_segments", 0)),
        )


@dataclass
class PhoneTemplateStore:
    phones: Dict[str, PhoneTemplate]
    n_channels: int
    feature_dim: int
    meta: dict[str, Any]

    def channel_profile(self, phone: str, n_channels: int) -> np.ndarray:
        """Non-negative channel routing vector summing to 1."""
        key = str(phone).strip().upper()
        tpl = self.phones.get(key)
        if tpl is None:
            return np.ones((int(n_channels),), dtype=np.float64) / float(n_channels)
        prof = np.asarray(tpl.channel_rms, dtype=np.float64).ravel()
        if prof.size != int(n_channels):
            prof = _resize_vector(prof, int(n_channels))
        prof = np.maximum(prof, 1e-9)
        return prof / float(np.sum(prof))

    def rate_scale(self, phone: str) -> float:
        key = str(phone).strip().upper()
        tpl = self.phones.get(key)
        return float(tpl.rate_scale) if tpl is not None else 1.0

    def duration_weight(self, phone: str) -> float:
        key = str(phone).strip().upper()
        tpl = self.phones.get(key)
        return float(tpl.duration_weight) if tpl is not None else 0.6

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "n_channels": int(self.n_channels),
            "feature_dim": int(self.feature_dim),
            "meta": dict(self.meta),
            "phones": {k: v.to_dict() for k, v in sorted(self.phones.items())},
        }


def _resize_vector(v: np.ndarray, n: int) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).ravel()
    if v.size == n:
        return v
    if v.size == 0:
        return np.ones((n,), dtype=np.float64)
    x_old = np.linspace(0.0, 1.0, v.size)
    x_new = np.linspace(0.0, 1.0, n)
    return np.interp(x_new, x_old, v)


def load_phone_templates(path: Path | str) -> PhoneTemplateStore:
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    phones = {str(k): PhoneTemplate.from_dict(v) for k, v in raw.get("phones", {}).items()}
    return PhoneTemplateStore(
        phones=phones,
        n_channels=int(raw.get("n_channels", 0)),
        feature_dim=int(raw.get("feature_dim", 0)),
        meta=dict(raw.get("meta", {})),
    )


def save_phone_templates(store: PhoneTemplateStore, path: Path | str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(store.to_dict(), indent=2), encoding="utf-8")
    return p


def blend_motor_weights_for_phone(
    mu_w: np.ndarray,
    mu_lab: np.ndarray,
    phone_inventory: Sequence[str],
    store: PhoneTemplateStore,
    *,
    blend: float = 0.65,
) -> None:
    """In-place blend of MU channel weights toward per-phone template profiles."""
    blend = float(np.clip(blend, 0.0, 1.0))
    n_ch = int(mu_w.shape[1])
    for lid, phone in enumerate(phone_inventory):
        prof = store.channel_profile(str(phone), n_ch).astype(np.float32)
        mask = mu_lab == int(lid)
        if not bool(mask.any()):
            continue
        for idx in np.where(mask)[0]:
            base = np.asarray(mu_w[int(idx)], dtype=np.float64)
            mixed = (1.0 - blend) * base + blend * prof
            mixed = np.maximum(mixed, 1e-9)
            mu_w[int(idx)] = (mixed / float(np.sum(mixed))).astype(np.float32)
