"""Physiological motor-unit parameters (fiber types, rate–size, recruitment order).

Literature basis (simplified for discrete-time synthesis):
- Henneman size principle: small (S) units recruited first, large (FF) last.
- Rate–size: smaller units reach lower max firing rates; FF units higher (De Luca).
- MUAP duration correlates inversely with fiber type (wider for S, narrower for FF).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Tuple

import numpy as np

from .muap import bipolar_muap_template, stretch_muap_template


class FiberType(IntEnum):
    S = 0  # slow oxidative (Type I)
    FR = 1  # fast resistant (Type IIa)
    FF = 2  # fast fatigable (Type IIx)


@dataclass(frozen=True)
class FiberTypeSpec:
    """Per-type physiology defaults (facial/neck sEMG scale)."""

    name: str
    recruitment_rank: float  # 0 = earliest, 1 = latest
    max_rate_hz: float
    muap_width_scale: float
    twitch_force_rel: float
    fraction_of_pool: float


FIBER_TYPE_SPECS: Tuple[FiberTypeSpec, ...] = (
    FiberTypeSpec("S", 0.15, 12.0, 1.28, 0.35, 0.40),
    FiberTypeSpec("FR", 0.50, 22.0, 1.00, 0.55, 0.38),
    FiberTypeSpec("FF", 0.85, 35.0, 0.78, 1.00, 0.22),
)


@dataclass
class MotorPoolPhysiology:
    """Per-unit physiology arrays (parallel to motor pool tensors)."""

    fiber_type: np.ndarray  # int8 (n_mu,)
    recruitment_rank: np.ndarray  # float32 (n_mu,) in [0, 1]
    max_rate_hz: np.ndarray  # float32 (n_mu,)
    refractory_samples: np.ndarray  # int32 (n_mu,)


def _assign_fiber_types(rng: np.random.Generator, n_mu: int) -> np.ndarray:
    """Sample fiber types from literature-ish pool fractions."""
    fracs = np.array([s.fraction_of_pool for s in FIBER_TYPE_SPECS], dtype=np.float64)
    fracs /= fracs.sum()
    return rng.choice(len(FIBER_TYPE_SPECS), size=n_mu, p=fracs).astype(np.int8)


def init_physiological_motor_pool(
    rng: np.random.Generator,
    n_motor_units: int,
    n_channels: int,
    n_labels: int,
    *,
    fs_hz: float,
    muap_duration_ms: float = 10.0,
    refractory_ms: float = 2.5,
    preset_channel_weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, MotorPoolPhysiology, List[np.ndarray]]:
    """Create motor pool with fiber-type ordering and physiology metadata.

    Returns ``mu_label_id``, ``mu_channel_weights``, ``mu_gain``, :class:`MotorPoolPhysiology`,
    and per-unit MUAP templates (width scaled by fiber type).
    """
    from .motor_pool import init_motor_unit_layer

    mu_lab, mu_w, mu_gain = init_motor_unit_layer(
        rng, n_motor_units, n_channels, n_labels, preset_channel_weights=preset_channel_weights
    )
    n_mu = int(n_motor_units)
    ftypes = _assign_fiber_types(rng, n_mu)

    ranks = np.zeros(n_mu, dtype=np.float32)
    max_rates = np.zeros(n_mu, dtype=np.float32)
    for ft in range(len(FIBER_TYPE_SPECS)):
        mask = ftypes == ft
        if not np.any(mask):
            continue
        spec = FIBER_TYPE_SPECS[ft]
        ranks[mask] = float(spec.recruitment_rank) + rng.normal(0.0, 0.04, size=int(mask.sum())).astype(
            np.float32
        )
        max_rates[mask] = float(spec.max_rate_hz) * rng.lognormal(0.0, 0.08, size=int(mask.sum())).astype(
            np.float32
        )

    # Size principle: gain inversely tracks recruitment rank (small units = lower gain).
    rank_norm = (ranks - ranks.min()) / (ranks.max() - ranks.min() + 1e-9)
    mu_gain = (1.15 - 0.55 * rank_norm).astype(np.float32)
    mu_gain *= rng.lognormal(0.0, 0.12, size=n_mu).astype(np.float32)
    mu_gain /= float(np.mean(mu_gain)) + 1e-12

    ref_samp = max(1, int(float(refractory_ms) * float(fs_hz) / 1000.0))
    refractory = np.full(n_mu, ref_samp, dtype=np.int32)

    phys = MotorPoolPhysiology(
        fiber_type=ftypes,
        recruitment_rank=ranks.astype(np.float32),
        max_rate_hz=max_rates.astype(np.float32),
        refractory_samples=refractory,
    )

    base_tpl = bipolar_muap_template(float(fs_hz), duration_ms=float(muap_duration_ms))
    templates: List[np.ndarray] = []
    for i in range(n_mu):
        ft = int(ftypes[i])
        ws = float(FIBER_TYPE_SPECS[ft].muap_width_scale)
        templates.append(stretch_muap_template(base_tpl, width_scale=ws))

    return mu_lab, mu_w, mu_gain, phys, templates


def cap_rates_by_physiology(rates_hz: np.ndarray, phys: MotorPoolPhysiology) -> np.ndarray:
    """Clip per-MU Poisson rates to fiber-type max rates."""
    r = np.asarray(rates_hz, dtype=np.float64).copy()
    np.minimum(r, phys.max_rate_hz.astype(np.float64), out=r)
    return np.maximum(r, 0.0)
