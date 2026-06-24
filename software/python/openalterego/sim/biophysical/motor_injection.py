"""Motor-pool MUAP synthesis into a time window (shared by word and phoneme timelines)."""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from .conduction import channel_delays_from_pickup
from .motor_pool import firing_rates_baseline_segment, firing_rates_token_segment
from .physiology import MotorPoolPhysiology, cap_rates_by_physiology
from .pool import add_muap_spikes
from .pool_batch import superpose_motor_pool_window
from .pool_fast import MotorPoolSynthCache, superpose_motor_pool_fast
from .recruitment import recruitment_rate_multipliers


class MotorPoolInjector:
    """Stateless helper bound to one stream's motor pool tensors."""

    __slots__ = ("cfg", "rng", "mu_lab", "mu_w", "mu_gain", "mu_tpl", "phys", "use_batch", "_unit_delays", "_cache", "_synth_mode")

    def __init__(
        self,
        cfg: Any,
        rng: np.random.Generator,
        mu_lab: np.ndarray,
        mu_w: np.ndarray,
        mu_gain: np.ndarray,
        mu_tpl: List[np.ndarray],
        *,
        phys: Optional[MotorPoolPhysiology] = None,
        use_batch: bool = True,
        synth_mode: str = "fast",
    ) -> None:
        self.cfg = cfg
        self.rng = rng
        self.mu_lab = mu_lab
        self.mu_w = mu_w
        self.mu_gain = mu_gain
        self.mu_tpl = mu_tpl
        self.phys = phys
        self.use_batch = bool(use_batch)
        self._synth_mode = str(synth_mode).strip().lower() or "fast"
        self._unit_delays: Optional[np.ndarray] = None
        if bool(cfg.use_conduction_delays) and bool(cfg.muap_spatial_spread):
            n_mu = int(mu_lab.size)
            c = int(mu_w.shape[1])
            fs = float(cfg.fs_hz)
            dlys = np.zeros((n_mu, c), dtype=np.int32)
            for i in range(n_mu):
                d = channel_delays_from_pickup(
                    mu_w[i],
                    fs,
                    rng,
                    velocity_m_s=float(cfg.conduction_velocity_m_s),
                    array_span_m=float(cfg.conduction_array_span_m),
                    jitter_samples=int(cfg.conduction_jitter_samples),
                )
                if d is not None:
                    dlys[i, :] = d
            self._unit_delays = dlys
        self._cache = MotorPoolSynthCache.from_pool(mu_tpl, mu_w, self._unit_delays)

    def emit_token_window(
        self,
        x_win: np.ndarray,
        fs: float,
        env: np.ndarray,
        motor_lid: int,
        rate_mod: float,
        jit: float,
        spread_motor: bool,
    ) -> None:
        if x_win.size == 0:
            return
        r_seg = firing_rates_token_segment(
            self.mu_lab,
            self.mu_gain,
            active_label_id=int(motor_lid),
            token_firing_rate_hz=float(self.cfg.token_firing_rate_hz),
            baseline_firing_rate_hz=float(self.cfg.baseline_firing_rate_hz),
            off_label_rate_scale=float(self.cfg.off_label_rate_scale),
        )
        r_seg = r_seg * rate_mod
        if self.cfg.use_recruitment:
            coup = float(self.cfg.recruitment_envelope_coupling)
            act = float(np.mean(env)) * coup + (1.0 - coup) * float(
                self.cfg.recruitment_baseline_activation
            )
            rank = self.phys.recruitment_rank if self.phys is not None else None
            r_seg = r_seg * recruitment_rate_multipliers(
                self.mu_gain,
                act,
                steepness=float(self.cfg.recruitment_steepness),
                recruitment_rank=rank,
            )
        if self.phys is not None:
            r_seg = cap_rates_by_physiology(r_seg, self.phys)
        self._emit_window(
            x_win, fs, r_seg, float(self.cfg.muap_amplitude_uV) * float(self.cfg.token_spatial_gain),
            env, jit, spread_motor, baseline_scale=1.0,
        )

    def emit_rest_window(
        self,
        x_win: np.ndarray,
        fs: float,
        rate_mod: float,
        jit: float,
        spread_motor: bool,
    ) -> None:
        if x_win.size == 0:
            return
        r_rest = firing_rates_baseline_segment(self.mu_gain, float(self.cfg.baseline_firing_rate_hz))
        r_rest = r_rest * rate_mod
        if self.cfg.use_recruitment:
            act_b = float(self.cfg.recruitment_baseline_activation)
            rank = self.phys.recruitment_rank if self.phys is not None else None
            r_rest = r_rest * recruitment_rate_multipliers(
                self.mu_gain,
                act_b,
                steepness=float(self.cfg.recruitment_steepness),
                recruitment_rank=rank,
            )
        if self.phys is not None:
            r_rest = cap_rates_by_physiology(r_rest, self.phys)
        bas = float(self.cfg.baseline_amplitude_scale)
        self._emit_window(
            x_win, fs, r_rest, float(self.cfg.muap_amplitude_uV) * bas,
            None, jit, spread_motor, baseline_scale=1.0,
        )

    def _emit_window(
        self,
        x_win: np.ndarray,
        fs: float,
        rates: np.ndarray,
        amp_scale: float,
        envelope: Optional[np.ndarray],
        jit: float,
        spread_motor: bool,
        *,
        baseline_scale: float,
    ) -> None:
        n_mu = int(self.mu_lab.size)
        amps = (float(amp_scale) * self.mu_gain.astype(np.float64)).astype(np.float64)
        spread_motor = bool(spread_motor)
        if self.use_batch and self._synth_mode in ("fast", "numba", "rust"):
            backend = self._synth_mode if self._synth_mode in ("numba", "rust") else "auto"
            superpose_motor_pool_fast(
                x_win,
                fs,
                self.rng,
                rates,
                amps,
                self._cache,
                envelope=envelope,
                time_jitter_std_s=jit,
                refractory_samples=self.phys.refractory_samples if self.phys is not None else None,
                spread_across_channels=spread_motor,
                prefer_convolve=self._unit_delays is None,
                backend=backend,
            )
            return
        if self.use_batch:
            superpose_motor_pool_window(
                x_win,
                fs,
                self.rng,
                rates,
                amps,
                self.mu_w,
                self.mu_tpl,
                envelope=envelope,
                time_jitter_std_s=jit,
                refractory_samples=self.phys.refractory_samples if self.phys is not None else None,
                spread_across_channels=spread_motor,
                channel_delays=self._unit_delays if spread_motor else None,
            )
            return
        for i in range(n_mu):
            dlys = None if self._unit_delays is None else self._unit_delays[i]
            add_muap_spikes(
                x_win,
                fs,
                self.rng,
                float(rates[i]),
                self.mu_w[i],
                self.mu_tpl[i],
                float(amps[i]),
                envelope=envelope,
                time_jitter_std_s=jit,
                spread_across_channels=spread_motor,
                channel_delay_samples=dlys,
            )

    def _delays_for_unit(self, i: int, fs: float, spread_motor: bool) -> Optional[np.ndarray]:
        if not spread_motor or not self.cfg.use_conduction_delays:
            return None
        return channel_delays_from_pickup(
            self.mu_w[i],
            fs,
            self.rng,
            velocity_m_s=float(self.cfg.conduction_velocity_m_s),
            array_span_m=float(self.cfg.conduction_array_span_m),
            jitter_samples=int(self.cfg.conduction_jitter_samples),
        )
