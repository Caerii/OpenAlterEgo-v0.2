"""Pre-motor sensor noise: white floor, AR(1), slow 'pink' AR, mains (+ harmonics), drift, motion bursts."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..realism import realism_preset_params
from .noise_fast import _ar1_chunk, apply_motion_burst_vectorized


class SensorNoiseState:
    """Mutable state carried across biophysical (and heuristic) chunks."""

    __slots__ = (
        "drift",
        "ar1_state",
        "line_phases",
        "line_phases_h2",
        "pink_ar1",
        "motion_left",
        "motion_total",
        "motion_amp",
        "motion_freq_hz",
        "motion_phase",
        "contact_dc",
        "lf_post_bp",
    )

    def __init__(self, n_channels: int, rng: np.random.Generator) -> None:
        c = int(n_channels)
        self.drift = np.zeros((c,), dtype=np.float32)
        self.ar1_state = np.zeros((c,), dtype=np.float32)
        self.line_phases = rng.uniform(0.0, 2.0 * math.pi, size=(c,)).astype(np.float32)
        self.line_phases_h2 = rng.uniform(0.0, 2.0 * math.pi, size=(c,)).astype(np.float32)
        self.pink_ar1 = np.zeros((c,), dtype=np.float32)
        self.motion_left = 0
        self.motion_total = 0
        self.motion_amp = 0.0
        self.motion_freq_hz = 2.5
        self.motion_phase = 0.0
        self.contact_dc = np.zeros((c,), dtype=np.float32)
        self.lf_post_bp = np.zeros((c,), dtype=np.float32)

    def apply_chunk(
        self,
        x: np.ndarray,
        cfg: Any,
        rng: np.random.Generator,
        *,
        chunk_start: int,
        fs: float,
        noise_scale: float,
    ) -> None:
        """Add sensor noise in-place; ``x`` is (n_samples, n_channels)."""
        n, c = x.shape
        ns = float(noise_scale)
        sigma_w = float(cfg.electrode_noise_uV) * ns
        preset_name = str(getattr(cfg, "realism_preset", "off"))
        rp = realism_preset_params(preset_name)

        x += rng.standard_normal(size=(n, c)).astype(np.float32) * sigma_w

        phi = float(np.clip(float(cfg.ar1_phi), 0.0, 0.999))
        innov_main = sigma_w * float(cfg.ar1_innovation_scale)
        pink_phi = float(np.clip(rp.pink_phi, 0.0, 0.999))
        pink_scale = float(rp.pink_innov_scale)
        pink_innov = sigma_w * pink_scale

        if innov_main > 0.0:
            innov = rng.normal(0.0, innov_main, size=(n, c)).astype(np.float32)
            ar1_out, self.ar1_state = _ar1_chunk(innov, phi, self.ar1_state)
            x += ar1_out
        if pink_phi > 0.0 and pink_scale > 0.0:
            innov_p = rng.normal(0.0, pink_innov, size=(n, c)).astype(np.float32)
            pink_out, self.pink_ar1 = _ar1_chunk(innov_p, pink_phi, self.pink_ar1)
            x += pink_out

        if float(cfg.line_noise_uV) > 0.0:
            f0 = float(cfg.mains_freq_hz)
            w = 2.0 * math.pi * f0 / fs
            amp_ln = float(cfg.line_noise_uV) * ns
            h2 = float(rp.mains_h2_relative)
            t = (float(chunk_start) + np.arange(n, dtype=np.float64)) / fs
            ph = w * t
            s1 = np.sin(ph[:, None] + self.line_phases[None, :])
            row = amp_ln * s1
            if h2 > 0.0:
                row = row + (amp_ln * h2) * np.sin(2.0 * ph[:, None] + self.line_phases_h2[None, :])
            x += row.astype(np.float32)

        if rp.motion_chunk_prob > 0.0 and self.motion_left <= 0:
            if float(rng.random()) < float(rp.motion_chunk_prob):
                lo, hi = float(rp.motion_len_s[0]), float(rp.motion_len_s[1])
                lo_i = max(1, int(lo * fs))
                hi_i = max(lo_i + 1, int(hi * fs) + 1)
                self.motion_left = int(rng.integers(lo_i, hi_i))
                self.motion_total = max(1, self.motion_left)
                self.motion_amp = float(
                    rng.lognormal(float(rp.motion_amp_log_mean), float(rp.motion_amp_log_sigma))
                ) * ns * float(getattr(cfg, "motion_burst_scale", 1.0))
                f_lo, f_hi = float(rp.motion_freq_hz[0]), float(rp.motion_freq_hz[1])
                self.motion_freq_hz = float(rng.uniform(f_lo, f_hi))
                self.motion_phase = float(rng.uniform(0.0, 2.0 * math.pi))

        if self.motion_left > 0:
            ch_w = rng.uniform(0.55, 1.0, size=(c,)).astype(np.float32)
            self.motion_left = apply_motion_burst_vectorized(
                x,
                chunk_start=chunk_start,
                fs=fs,
                motion_left=self.motion_left,
                motion_total=self.motion_total,
                motion_amp=self.motion_amp,
                motion_freq_hz=self.motion_freq_hz,
                motion_phase=self.motion_phase,
                shared_fraction=float(rp.motion_shared_fraction),
                line_phases=self.line_phases,
                ch_weights=ch_w,
            )

        if float(rp.contact_step_prob) > 0.0 and float(rp.contact_dc_step_uV) > 0.0:
            if float(rng.random()) < float(rp.contact_step_prob):
                mask = rng.random(size=(c,)) < 0.55
                step = rng.normal(
                    0.0, float(rp.contact_dc_step_uV) * ns, size=(c,)
                ).astype(np.float32)
                self.contact_dc = np.where(mask, self.contact_dc + step, self.contact_dc)
            decay = math.exp(-float(n) / max(1.0, 8.0 * fs))
            self.contact_dc *= float(decay)
            x += self.contact_dc[None, :]

        drift_step = float(cfg.drift_uV_per_s) * ns / math.sqrt(fs)
        self.drift += rng.normal(scale=drift_step, size=(c,)).astype(np.float32)
        x += self.drift[None, :]

    def apply_post_bp_lf(
        self,
        x: np.ndarray,
        cfg: Any,
        rng: np.random.Generator,
        *,
        lf_snr_scale: float = 1.0,
    ) -> None:
        """Add LF pink AR(1) after bandpass so SNR/motion metrics match wearable literature."""
        preset_name = str(getattr(cfg, "realism_preset", "off"))
        rp = realism_preset_params(preset_name)
        base = float(rp.lf_snr_innov_uV) * float(lf_snr_scale)
        if base <= 0.0:
            return
        n, c = x.shape
        phi = float(np.clip(float(rp.lf_snr_phi), 0.0, 0.999))
        innov = rng.normal(0.0, float(base), size=(n, c)).astype(np.float32)
        lf_out, self.lf_post_bp = _ar1_chunk(innov, phi, self.lf_post_bp)
        x += lf_out
