"""Detect when live or offline SNR drift suggests re-calibration (Tang / SilentWear targets)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

# Literature consensus: re-calibrate when SNR drops ~3 dB vs baseline (roadmap §7).
DEFAULT_RECAL_WARN_DB = 3.0


@dataclass(frozen=True)
class RecalibrationConfig:
    baseline_snr_db: float
    warn_db: float = DEFAULT_RECAL_WARN_DB
    motion_index_warn: float = 0.40
    consecutive_low: int = 3
    cooldown_s: float = 30.0
    snr_ema_alpha: float = 0.15


@dataclass(frozen=True)
class RecalibrationStatus:
    snr_db: Optional[float]
    snr_ema: Optional[float]
    snr_deficit_db: float
    motion_index: float
    low_streak: int
    re_calibration_suggested: bool
    should_broadcast: bool
    reasons: List[str] = field(default_factory=list)

    def to_meta(self) -> dict:
        return {
            "snr_db": self.snr_db,
            "snr_ema": self.snr_ema,
            "snr_deficit_db": round(float(self.snr_deficit_db), 2),
            "motion_index": round(float(self.motion_index), 3),
            "re_calibration_suggested": bool(self.re_calibration_suggested),
            "recalibration_reasons": list(self.reasons),
        }


class RecalibrationMonitor:
    """Track SNR vs calibration baseline; suggest re-cal after sustained deficit."""

    __slots__ = ("cfg", "_snr_ema", "_low_streak", "_last_broadcast_t")

    def __init__(self, cfg: RecalibrationConfig) -> None:
        self.cfg = cfg
        self._snr_ema: Optional[float] = None
        self._low_streak = 0
        self._last_broadcast_t = 0.0

    @classmethod
    def from_baseline(
        cls,
        baseline_snr_db: float,
        *,
        warn_db: float = DEFAULT_RECAL_WARN_DB,
        cooldown_s: float = 30.0,
    ) -> RecalibrationMonitor:
        return cls(
            RecalibrationConfig(
                baseline_snr_db=float(baseline_snr_db),
                warn_db=float(warn_db),
                cooldown_s=float(cooldown_s),
            )
        )

    def reset(self) -> None:
        self._snr_ema = None
        self._low_streak = 0
        self._last_broadcast_t = 0.0

    def update(
        self,
        snr_db: Optional[float],
        *,
        motion_index: float = 0.0,
        now: Optional[float] = None,
    ) -> RecalibrationStatus:
        now = float(time.time() if now is None else now)
        reasons: List[str] = []
        deficit = 0.0

        if snr_db is not None:
            s = float(snr_db)
            if self._snr_ema is None:
                self._snr_ema = s
            else:
                a = float(self.cfg.snr_ema_alpha)
                self._snr_ema = (1.0 - a) * self._snr_ema + a * s
            deficit = max(0.0, float(self.cfg.baseline_snr_db) - self._snr_ema)
            if deficit >= float(self.cfg.warn_db):
                self._low_streak += 1
                reasons.append(f"snr_deficit_{deficit:.1f}db")
            else:
                self._low_streak = 0

        if float(motion_index) >= float(self.cfg.motion_index_warn):
            reasons.append(f"motion_index_{float(motion_index):.2f}")

        suggested = self._low_streak >= int(self.cfg.consecutive_low)
        if suggested and not reasons:
            reasons.append("sustained_low_snr")

        should_broadcast = suggested and (now - self._last_broadcast_t) >= float(self.cfg.cooldown_s)
        if should_broadcast:
            self._last_broadcast_t = now

        return RecalibrationStatus(
            snr_db=snr_db,
            snr_ema=self._snr_ema,
            snr_deficit_db=float(deficit),
            motion_index=float(motion_index),
            low_streak=int(self._low_streak),
            re_calibration_suggested=bool(suggested),
            should_broadcast=bool(should_broadcast),
            reasons=reasons,
        )


def assess_session_recalibration(
    *,
    session_snr_db: Optional[float],
    baseline_snr_db: Optional[float],
    motion_index: float = 0.0,
    warn_db: float = DEFAULT_RECAL_WARN_DB,
) -> RecalibrationStatus:
    """One-shot offline check for a recorded session vs profile baseline."""
    if baseline_snr_db is None or session_snr_db is None:
        return RecalibrationStatus(
            snr_db=session_snr_db,
            snr_ema=session_snr_db,
            snr_deficit_db=0.0,
            motion_index=float(motion_index),
            low_streak=0,
            re_calibration_suggested=False,
            should_broadcast=False,
            reasons=["missing_baseline_or_session_snr"],
        )
    mon = RecalibrationMonitor.from_baseline(float(baseline_snr_db), warn_db=float(warn_db), cooldown_s=0.0)
    st = mon.update(float(session_snr_db), motion_index=float(motion_index), now=0.0)
    suggested = float(session_snr_db) < float(baseline_snr_db) - float(warn_db)
    reasons = list(st.reasons)
    if suggested and "snr_deficit" not in " ".join(reasons):
        reasons.append(f"session_below_baseline_by_{float(baseline_snr_db) - float(session_snr_db):.1f}db")
    return RecalibrationStatus(
        snr_db=st.snr_db,
        snr_ema=st.snr_ema,
        snr_deficit_db=max(0.0, float(baseline_snr_db) - float(session_snr_db)),
        motion_index=st.motion_index,
        low_streak=1 if suggested else 0,
        re_calibration_suggested=bool(suggested),
        should_broadcast=bool(suggested),
        reasons=reasons,
    )
