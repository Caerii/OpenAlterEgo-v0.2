"""Wearable sEMG realism presets (noise morphology, motion, frontend imperfection).

These layers sit on top of the existing white + AR(1) + drift + mains model. They are **not**
a full biophysical volume conductor or motion IMU model; they reproduce common *qualitative*
failure modes seen in real acquisitions (correlated LF, harmonic hum, intermittent motion,
per-electrode gain/offset, mild saturation).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np

RealismPreset = Literal["off", "wearable", "tang", "field"]


@dataclass(frozen=True)
class RealismParams:
    """Resolved parameters for :class:`~openalterego.sim.biophysical.sensor_pipeline.SensorNoiseState`."""

    pink_phi: float
    pink_innov_scale: float
    mains_h2_relative: float
    motion_chunk_prob: float
    motion_len_s: Tuple[float, float]
    motion_amp_log_mean: float
    motion_amp_log_sigma: float
    motion_freq_hz: Tuple[float, float]
    channel_gain_log_sigma: float
    channel_dc_uV_half_range: float
    adc_soft_clip_uV: Optional[float]
    # Shared rigid motion vs per-channel residual (0=all independent, 1=all shared).
    motion_shared_fraction: float = 0.82
    # Intermittent electrode contact steps (dry/wearable impedance shifts).
    contact_step_prob: float = 0.0
    contact_dc_step_uV: float = 0.0
    # Post-bandpass LF contaminant (motion/drift visible in 0.5–5 Hz SNR metric).
    lf_snr_innov_uV: float = 0.0
    lf_snr_phi: float = 0.992


def realism_preset_params(preset: RealismPreset | str) -> RealismParams:
    """Map a preset name to concrete hyperparameters."""
    key = str(preset).strip().lower()
    if key in ("", "off", "none", "minimal"):
        return RealismParams(
            pink_phi=0.0,
            pink_innov_scale=0.0,
            mains_h2_relative=0.0,
            motion_chunk_prob=0.0,
            motion_len_s=(0.05, 0.2),
            motion_amp_log_mean=0.0,
            motion_amp_log_sigma=0.0,
            motion_freq_hz=(2.0, 4.0),
            channel_gain_log_sigma=0.0,
            channel_dc_uV_half_range=0.0,
            adc_soft_clip_uV=None,
        )
    if key == "wearable":
        return RealismParams(
            pink_phi=0.993,
            pink_innov_scale=0.45,
            mains_h2_relative=0.2,
            motion_chunk_prob=0.038,
            motion_len_s=(0.07, 0.58),
            motion_amp_log_mean=math.log(38.0),
            motion_amp_log_sigma=0.52,
            motion_freq_hz=(1.1, 5.8),
            channel_gain_log_sigma=0.075,
            channel_dc_uV_half_range=7.0,
            adc_soft_clip_uV=None,
            motion_shared_fraction=0.78,
            contact_step_prob=0.006,
            contact_dc_step_uV=12.0,
        )
    if key == "tang":
        # Tang et al. 2025 wearable textile: motion-heavy but SNR-calibratable via noise_scale.
        return RealismParams(
            pink_phi=0.994,
            pink_innov_scale=0.52,
            mains_h2_relative=0.24,
            motion_chunk_prob=0.058,
            motion_len_s=(0.10, 0.88),
            motion_amp_log_mean=math.log(46.0),
            motion_amp_log_sigma=0.50,
            motion_freq_hz=(0.85, 6.8),
            channel_gain_log_sigma=0.082,
            channel_dc_uV_half_range=9.0,
            adc_soft_clip_uV=None,
            motion_shared_fraction=0.88,
            contact_step_prob=0.014,
            contact_dc_step_uV=16.0,
            lf_snr_innov_uV=6.5,
            lf_snr_phi=0.993,
        )
    if key == "field":
        return RealismParams(
            pink_phi=0.995,
            pink_innov_scale=0.68,
            mains_h2_relative=0.32,
            motion_chunk_prob=0.085,
            motion_len_s=(0.12, 1.05),
            motion_amp_log_mean=math.log(62.0),
            motion_amp_log_sigma=0.62,
            motion_freq_hz=(0.75, 8.0),
            channel_gain_log_sigma=0.11,
            channel_dc_uV_half_range=14.0,
            adc_soft_clip_uV=980.0,
            motion_shared_fraction=0.90,
            contact_step_prob=0.022,
            contact_dc_step_uV=22.0,
            lf_snr_innov_uV=11.0,
            lf_snr_phi=0.994,
        )
    raise ValueError(f"unknown realism preset {preset!r} (use off, wearable, tang, field)")


def apply_frontend_imperfections(
    x,
    ch_gain,
    ch_dc,
    *,
    adc_soft_clip_uV: Optional[float],
) -> None:
    """Scale per channel, add DC offset, optional soft clip (in-place, ``x`` shape (n, c))."""
    x *= ch_gain
    x += ch_dc
    if adc_soft_clip_uV is not None and float(adc_soft_clip_uV) > 0:
        lim = float(adc_soft_clip_uV)
        x[...] = np.tanh(x / lim) * lim
