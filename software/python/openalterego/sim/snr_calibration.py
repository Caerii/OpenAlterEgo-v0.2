"""Auto-tune synthetic noise to literature SNR targets (Tang et al. 2025)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..dsp.quality import assess_signal_quality
from .biophysical.stream import BiophysicalSimStream, BiophysicalSimStreamConfig
from .literature import resolve_sim_token_band
from .stream import ScenarioConfig

# Tang et al. (2025) headphone textile EMG benchmarks (docs/12-references.md).
TANG_SNR_STATIC_DB = 18.9
TANG_SNR_MOTION_DB = 12.7


@dataclass(frozen=True)
class SnrCalibrationResult:
    lf_snr_scale: float
    measured_snr_db: Optional[float]
    measured_motion_index: float
    target_snr_db: float
    iterations: int

    @property
    def noise_scale(self) -> float:
        """Backward-compatible alias (maps to ``lf_snr_scale``)."""
        return float(self.lf_snr_scale)


@dataclass(frozen=True)
class SnrRegimeCalibrationResult:
    """Static + motion-regime SNR calibration (Tang 18.9 / 12.7 dB)."""

    lf_snr_scale: float
    motion_burst_scale: float
    static_snr_db: Optional[float]
    motion_snr_db: Optional[float]
    measured_motion_index: float
    static_iterations: int
    motion_iterations: int

    def to_dict(self) -> dict:
        return {
            "lf_snr_scale": float(self.lf_snr_scale),
            "motion_burst_scale": float(self.motion_burst_scale),
            "static_snr_db": self.static_snr_db,
            "motion_snr_db": self.motion_snr_db,
            "measured_motion_index": float(self.measured_motion_index),
            "static_iterations": int(self.static_iterations),
            "motion_iterations": int(self.motion_iterations),
        }


def _run_probe(
    *,
    fs_hz: int,
    channels: int,
    emg_paradigm: str,
    realism_preset: str,
    noise_scale: float,
    electrode_noise_uV: float,
    line_noise_uV: float,
    drift_uV_per_s: float,
    crosstalk: float,
    montage_name: Optional[str],
    seed: int,
    duration_s: float,
    lf_snr_scale: float = 1.0,
    motion_burst_scale: float = 1.0,
) -> Tuple[Optional[float], float, np.ndarray, np.ndarray]:
    sc = ScenarioConfig(labels=["yes", "no"], p_event=0.72)
    cfg = BiophysicalSimStreamConfig(
        fs_hz=int(fs_hz),
        channels=int(channels),
        chunk_ms=40,
        seed=int(seed),
        scenario=sc,
        realtime_clock=False,
        emg_paradigm=str(emg_paradigm),
        electrode_noise_uV=float(electrode_noise_uV),
        drift_uV_per_s=float(drift_uV_per_s),
        crosstalk=float(crosstalk),
        line_noise_uV=float(line_noise_uV),
        noise_scale=float(noise_scale),
        lf_snr_scale=float(lf_snr_scale),
        motion_burst_scale=float(motion_burst_scale),
        realism_preset=realism_preset,  # type: ignore[arg-type]
        montage_name=montage_name,
    )
    sim = BiophysicalSimStream(cfg)
    target = max(1, int(float(duration_s) * int(fs_hz)))
    buf = np.zeros((target, int(channels)), dtype=np.float32)
    motion_mask = np.zeros((target,), dtype=bool)
    i = 0
    while i < target:
        in_motion = int(sim._sensor.motion_left) > 0  # noqa: SLF001
        ch = sim.next_chunk()
        n = int(ch.samples.shape[0])
        take = min(n, target - i)
        buf[i : i + take, :] = ch.samples[:take, :]
        if in_motion:
            motion_mask[i : i + take] = True
        i += take
    band = resolve_sim_token_band(int(fs_hz), str(emg_paradigm), None)
    q = assess_signal_quality(
        buf,
        fs_hz=float(fs_hz),
        signal_band_hz=band,
        noise_band_hz=(0.5, 5.0),
        low_freq_cutoff_hz=5.0,
        axis=0,
    )
    return q.snr_db, float(q.motion_index), buf, motion_mask


def _assess_regime_snr(
    buf: np.ndarray,
    mask: np.ndarray,
    *,
    fs_hz: int,
    emg_paradigm: str,
) -> Optional[float]:
    if not bool(mask.any()):
        return None
    band = resolve_sim_token_band(int(fs_hz), str(emg_paradigm), None)
    q = assess_signal_quality(
        buf[mask, :],
        fs_hz=float(fs_hz),
        signal_band_hz=band,
        noise_band_hz=(0.5, 5.0),
        low_freq_cutoff_hz=5.0,
        axis=0,
    )
    return q.snr_db


def tune_noise_scale(
    *,
    fs_hz: int,
    channels: int,
    emg_paradigm: str = "semg_literature_clamped",
    realism_preset: str = "tang",
    target_snr_db: float = TANG_SNR_STATIC_DB,
    electrode_noise_uV: float = 22.0,
    line_noise_uV: float = 8.0,
    drift_uV_per_s: float = 18.0,
    crosstalk: float = 0.12,
    noise_scale: float = 1.0,
    montage_name: Optional[str] = None,
    seed: int = 42,
    probe_duration_s: float = 6.0,
    tol_db: float = 1.2,
    max_iter: int = 12,
) -> SnrCalibrationResult:
    """Binary-search ``lf_snr_scale`` so assessed session SNR approaches ``target_snr_db``."""
    lo, hi = 0.005, 5.0
    best_scale = 1.0
    best_snr: Optional[float] = None
    best_motion = 0.0
    iterations = 0

    for iterations in range(1, int(max_iter) + 1):
        mid = 0.5 * (lo + hi)
        snr, motion, _, _ = _run_probe(
            fs_hz=fs_hz,
            channels=channels,
            emg_paradigm=emg_paradigm,
            realism_preset=realism_preset,
            noise_scale=noise_scale,
            electrode_noise_uV=electrode_noise_uV,
            line_noise_uV=line_noise_uV,
            drift_uV_per_s=drift_uV_per_s,
            crosstalk=crosstalk,
            montage_name=montage_name,
            seed=seed,
            duration_s=probe_duration_s,
            lf_snr_scale=mid,
        )
        if snr is None:
            break
        best_scale, best_snr, best_motion = mid, snr, motion
        if abs(float(snr) - float(target_snr_db)) <= float(tol_db):
            break
        if float(snr) > float(target_snr_db):
            lo = mid
        else:
            hi = mid

    return SnrCalibrationResult(
        lf_snr_scale=float(best_scale),
        measured_snr_db=best_snr,
        measured_motion_index=float(best_motion),
        target_snr_db=float(target_snr_db),
        iterations=int(iterations),
    )


def tune_snr_regimes(
    *,
    fs_hz: int,
    channels: int,
    emg_paradigm: str = "semg_literature_clamped",
    realism_preset: str = "tang",
    static_target_db: float = TANG_SNR_STATIC_DB,
    motion_target_db: float = TANG_SNR_MOTION_DB,
    electrode_noise_uV: float = 22.0,
    line_noise_uV: float = 8.0,
    drift_uV_per_s: float = 18.0,
    crosstalk: float = 0.12,
    noise_scale: float = 1.0,
    montage_name: Optional[str] = None,
    seed: int = 42,
    probe_duration_s: float = 10.0,
    tol_db: float = 1.5,
    max_iter: int = 12,
) -> SnrRegimeCalibrationResult:
    """Calibrate static SNR (``lf_snr_scale``) then motion-burst SNR (``motion_burst_scale``)."""
    static = tune_noise_scale(
        fs_hz=fs_hz,
        channels=channels,
        emg_paradigm=emg_paradigm,
        realism_preset=realism_preset,
        target_snr_db=float(static_target_db),
        electrode_noise_uV=electrode_noise_uV,
        line_noise_uV=line_noise_uV,
        drift_uV_per_s=drift_uV_per_s,
        crosstalk=crosstalk,
        noise_scale=noise_scale,
        montage_name=montage_name,
        seed=seed,
        probe_duration_s=probe_duration_s,
        tol_db=tol_db,
        max_iter=max_iter,
    )

    lo, hi = 0.05, 8.0
    best_motion_scale = 1.0
    best_motion_snr: Optional[float] = None
    best_motion_index = float(static.measured_motion_index)
    motion_iters = 0
    last_buf: Optional[np.ndarray] = None
    last_mask: Optional[np.ndarray] = None

    for motion_iters in range(1, int(max_iter) + 1):
        mid = 0.5 * (lo + hi)
        _, motion_index, buf, mask = _run_probe(
            fs_hz=fs_hz,
            channels=channels,
            emg_paradigm=emg_paradigm,
            realism_preset=realism_preset,
            noise_scale=noise_scale,
            electrode_noise_uV=electrode_noise_uV,
            line_noise_uV=line_noise_uV,
            drift_uV_per_s=drift_uV_per_s,
            crosstalk=crosstalk,
            montage_name=montage_name,
            seed=seed + 17,
            duration_s=max(float(probe_duration_s), 12.0),
            lf_snr_scale=float(static.lf_snr_scale),
            motion_burst_scale=float(mid),
        )
        last_buf, last_mask = buf, mask
        best_motion_index = float(motion_index)
        motion_snr = _assess_regime_snr(buf, mask, fs_hz=fs_hz, emg_paradigm=emg_paradigm)
        if motion_snr is None:
            break
        best_motion_scale, best_motion_snr = mid, motion_snr
        if abs(float(motion_snr) - float(motion_target_db)) <= float(tol_db):
            break
        if float(motion_snr) > float(motion_target_db):
            lo = mid
        else:
            hi = mid

    static_snr = static.measured_snr_db
    if last_buf is not None and last_mask is not None and bool((~last_mask).any()):
        static_snr = _assess_regime_snr(
            last_buf,
            ~last_mask,
            fs_hz=fs_hz,
            emg_paradigm=emg_paradigm,
        )

    return SnrRegimeCalibrationResult(
        lf_snr_scale=float(static.lf_snr_scale),
        motion_burst_scale=float(best_motion_scale),
        static_snr_db=static_snr,
        motion_snr_db=best_motion_snr,
        measured_motion_index=float(best_motion_index),
        static_iterations=int(static.iterations),
        motion_iterations=int(motion_iters),
    )
