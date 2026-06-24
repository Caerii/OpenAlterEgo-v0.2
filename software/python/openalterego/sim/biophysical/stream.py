"""Chunked biophysical-style sEMG stream (MUAP superposition + scenario timeline)."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

from ..realism import RealismPreset, apply_frontend_imperfections, realism_preset_params

import numpy as np
from scipy import signal

from ...core.types import FrameChunk
from ..literature import (
    DEFAULT_AR1_INNOVATION_SCALE,
    LITERATURE_MODEL_VERSION,
    resolve_sim_token_band,
)
from ..phonology import (
    PhonemeSegment,
    expand_word_to_phones,
    iter_phone_slices,
    merge_lexicon,
    partition_event_to_phones,
    partition_phones_in_event,
    phone_inventory,
)
from ..phonology.coarticulation import (
    build_phone_coarticulation_envelopes,
    iter_coarticulated_phone_jobs,
)
from ..phonology.templates import blend_motor_weights_for_phone, load_phone_templates
from ..stream import ScenarioConfig, ScriptedWordEvent, SimEvent, _envelope, _make_profiles, _sample_choice
from .forward_model import green_pickup_matrix, motor_unit_pickup_weights
from .motor_injection import MotorPoolInjector
from .motor_pool import init_motor_unit_layer
from .physiology import MotorPoolPhysiology, init_physiological_motor_pool
from .muap import bipolar_muap_template, stretch_muap_template
from .pool import add_muap_spikes
from .sensor_pipeline import SensorNoiseState
from .versions import BIOPHYS_MODEL_VERSION, BIOPHYS_MODEL_VERSION_LEGACY
from .volume import diffuse_electrode_mix


@dataclass
class BiophysicalSimStreamConfig:
    fs_hz: int = 250
    channels: int = 8
    chunk_ms: int = 40
    seed: int = 1337
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    realtime_clock: bool = True

    emg_paradigm: str = "semg_literature_clamped"
    token_band_hz: Optional[Tuple[float, float]] = None

    electrode_noise_uV: float = 14.0
    drift_uV_per_s: float = 16.0
    # Neighbor weight in "neighbor" mix; lower bound for diffuse mix neighbor term.
    crosstalk: float = 0.12
    # Sensor / environment (Tang-style LF correlation, optional mains; see SimStream).
    ar1_phi: float = 0.97
    ar1_innovation_scale: float = DEFAULT_AR1_INNOVATION_SCALE
    line_noise_uV: float = 1.5
    mains_freq_hz: float = 60.0
    noise_scale: float = 1.0  # scales white, AR(1), drift, line (SNR regime sweeps)
    lf_snr_scale: float = 1.0  # scales post-BP LF contaminant (Tang SNR calibration)
    motion_burst_scale: float = 1.0  # scales motion-burst amplitude (Tang motion SNR)
    volume_mix_mode: Literal["neighbor", "diffuse"] = "diffuse"
    band_limit_output: bool = True  # causal IIR bandpass, state carried across chunks
    muap_spatial_spread: bool = True  # motor pool: MUAP on all ch. weighted by routing
    muap_width_jitter_log_sigma: float = 0.10
    common_drive_std: float = 0.14
    common_drive_ar1_phi: float = 0.90

    baseline_firing_rate_hz: float = 14.0
    token_firing_rate_hz: float = 72.0
    muap_duration_ms: float = 10.0
    muap_amplitude_uV: float = 55.0
    token_spatial_gain: float = 1.0
    baseline_amplitude_scale: float = 0.82

    use_motor_unit_pool: bool = True
    n_motor_units: int = 48
    # Fiber-type pool (S/FR/FF), rate–size, refractory; batch vectorized synthesis.
    use_physiological_pool: bool = True
    refractory_ms: float = 2.5
    use_batch_synthesis: bool = True
    synth_mode: Literal["fast", "batch", "legacy", "numba", "rust"] = "fast"
    off_label_rate_scale: float = 0.22
    spike_jitter_std_s: float = 0.0012

    use_forward_pickup: bool = True
    n_muscle_sources: int = 14
    forward_falloff: float = 0.17
    use_conduction_delays: bool = True
    conduction_velocity_m_s: float = 4.0
    conduction_array_span_m: float = 0.042
    conduction_jitter_samples: int = 1
    use_recruitment: bool = True
    recruitment_envelope_coupling: float = 0.88
    recruitment_baseline_activation: float = 0.34
    recruitment_steepness: float = 14.0
    # Extra sensor/motion morphology + frontend gain/offset (see sim.realism).
    realism_preset: RealismPreset = "wearable"
    # When set, forward pickup uses montage site geometry (hardware DSL electrodes.montage).
    montage_name: Optional[str] = None
    # Per-phone templates (channel RMS + rate scale) from ``analyze fit-phone-templates``.
    phone_templates_path: Optional[str] = None
    # Raised-cosine overlap between adjacent phones (phoneme drive).
    coarticulation_enabled: bool = True
    coarticulation_overlap_fraction: float = 0.28
    coarticulation_min_overlap_ms: float = 10.0


class BiophysicalSimStream:
    """State machine aligned with :class:`~openalterego.sim.stream.SimStream`; signal from MUAP pool."""

    def __init__(self, config: BiophysicalSimStreamConfig):
        self.cfg = config
        self.rng = np.random.default_rng(int(config.seed))
        self.chunk_samples = max(1, int(config.fs_hz * config.chunk_ms / 1000))

        token_band = resolve_sim_token_band(
            int(config.fs_hz),
            str(config.emg_paradigm),
            config.token_band_hz,
        )
        self._resolved_token_band = token_band
        self._lex = merge_lexicon(config.scenario.phone_lexicon)
        self._phone_template_store = None
        if config.phone_templates_path:
            self._phone_template_store = load_phone_templates(str(config.phone_templates_path))
        self._use_phoneme_drive = bool(config.use_motor_unit_pool) and str(
            config.scenario.drive_mode
        ) == "phoneme"
        if self._use_phoneme_drive:
            self._phone_inventory = phone_inventory(config.scenario.labels, self._lex)
            self._pid = {str(p): i for i, p in enumerate(self._phone_inventory)}
            n_lab_motor = len(self._phone_inventory)
        else:
            self._phone_inventory = [str(x) for x in config.scenario.labels]
            self._pid = {}
            n_lab_motor = len(config.scenario.labels)

        self.profiles = _make_profiles(
            config.scenario.labels,
            config.channels,
            seed=int(config.seed),
            amplitude_uV=1.0,
            band_hz=token_band,
        )
        self._muap = bipolar_muap_template(
            float(config.fs_hz),
            duration_ms=float(config.muap_duration_ms),
        )

        self._lid: Dict[str, int] = {str(lab): i for i, lab in enumerate(config.scenario.labels)}
        self._forward_G: Optional[np.ndarray] = None
        self._physiology: Optional[MotorPoolPhysiology] = None
        if config.use_motor_unit_pool:
            n_mu = int(config.n_motor_units)
            n_lab = int(n_lab_motor)
            c_ch = int(config.channels)
            w_pick = None
            if config.use_forward_pickup:
                from ..montage_geometry import resolve_forward_pickup_matrix

                self._forward_G = resolve_forward_pickup_matrix(
                    c_ch,
                    int(config.n_muscle_sources),
                    montage_name=config.montage_name,
                    falloff=float(config.forward_falloff),
                )
                _, w_pick = motor_unit_pickup_weights(self.rng, self._forward_G, n_mu)
            if bool(config.use_physiological_pool):
                self._mu_lab, self._mu_w, self._mu_gain, self._physiology, self._mu_tpl = (
                    init_physiological_motor_pool(
                        self.rng,
                        n_mu,
                        c_ch,
                        n_lab,
                        fs_hz=float(config.fs_hz),
                        muap_duration_ms=float(config.muap_duration_ms),
                        refractory_ms=float(config.refractory_ms),
                        preset_channel_weights=w_pick,
                    )
                )
            else:
                if w_pick is not None:
                    self._mu_lab, self._mu_w, self._mu_gain = init_motor_unit_layer(
                        self.rng, n_mu, c_ch, n_lab, preset_channel_weights=w_pick
                    )
                else:
                    self._mu_lab, self._mu_w, self._mu_gain = init_motor_unit_layer(
                        self.rng, n_mu, c_ch, n_lab
                    )
                self._mu_tpl = []
        else:
            self._mu_lab = None
            self._mu_w = None
            self._mu_gain = None

        if (
            self._phone_template_store is not None
            and self._mu_lab is not None
            and self._mu_w is not None
            and self._use_phoneme_drive
        ):
            blend_motor_weights_for_phone(
                self._mu_w,
                self._mu_lab,
                self._phone_inventory,
                self._phone_template_store,
            )

        if config.use_motor_unit_pool and not bool(config.use_physiological_pool):
            sig = float(config.muap_width_jitter_log_sigma)
            base_tpl = self._muap
            n_mu = int(config.n_motor_units)
            self._mu_tpl = [
                stretch_muap_template(base_tpl, width_scale=float(self.rng.lognormal(0.0, sig)))
                for _ in range(n_mu)
            ]
        elif not config.use_motor_unit_pool:
            self._mu_tpl = []

        self.sample_index = 0
        self._t0_wall = time.time()
        self._active_label: Optional[str] = None
        self._active_remaining = 0
        self._phone_seq: List[str] = []
        self._phone_seg_lens: List[int] = []
        self._phone_coart_env: Optional[np.ndarray] = None

        sched = self.cfg.scenario.scripted_schedule
        self._scripted_queue: List[ScriptedWordEvent] = list(sched) if sched else []
        self._last_scripted_word_idx: Optional[int] = None
        self._pending_trial_id: Optional[int] = None
        self._pending_word_idx: Optional[int] = None
        self._gap_remaining = 0 if self._scripted_queue else self._sample_gap()
        self._current_event_start: Optional[int] = None
        self._event_len = 0

        self.events: List[SimEvent] = []
        self.phoneme_events: List[PhonemeSegment] = []

        self._sensor = SensorNoiseState(int(config.channels), self.rng)
        self._motor_injector: Optional[MotorPoolInjector] = None
        if config.use_motor_unit_pool and self._mu_lab is not None:
            self._motor_injector = MotorPoolInjector(
                config,
                self.rng,
                self._mu_lab,
                self._mu_w,
                self._mu_gain,
                self._mu_tpl,
                phys=self._physiology,
                use_batch=bool(config.use_batch_synthesis),
                synth_mode=str(config.synth_mode),
            )
        self._common_drive = 0.0

        c = int(config.channels)
        ct = float(config.crosstalk)
        if str(config.volume_mix_mode) == "diffuse":
            nw = max(0.06, ct)
            self._mix = diffuse_electrode_mix(c, self.rng, neighbor_weight=nw)
        elif ct > 0.0:
            self._mix = np.eye(c, dtype=np.float32)
            for i in range(c):
                if i - 1 >= 0:
                    self._mix[i, i - 1] += ct
                if i + 1 < c:
                    self._mix[i, i + 1] += ct
            self._mix = (self._mix / np.sum(self._mix, axis=1, keepdims=True)).astype(np.float32)
        else:
            self._mix = np.eye(c, dtype=np.float32)

        brp = realism_preset_params(str(config.realism_preset))
        if brp.channel_gain_log_sigma > 0.0:
            self._ch_gain = self.rng.lognormal(
                0.0, float(brp.channel_gain_log_sigma), size=(c,)
            ).astype(np.float32)
        else:
            self._ch_gain = np.ones((c,), dtype=np.float32)
        if brp.channel_dc_uV_half_range > 0.0:
            half = float(brp.channel_dc_uV_half_range)
            self._ch_dc = self.rng.uniform(-half, half, size=(c,)).astype(np.float32)
        else:
            self._ch_dc = np.zeros((c,), dtype=np.float32)
        self._adc_clip_uV: Optional[float] = (
            float(brp.adc_soft_clip_uV) if brp.adc_soft_clip_uV is not None else None
        )

        self._bp_sos: Optional[np.ndarray] = None
        self._bp_zi: Optional[np.ndarray] = None
        if config.band_limit_output:
            fs0 = float(config.fs_hz)
            lo, hi = float(token_band[0]), float(token_band[1])
            lo = max(lo, fs0 * 0.004)
            hi = min(hi, fs0 * 0.49)
            if hi > lo + 1.0:
                self._bp_sos = signal.butter(4, [lo, hi], btype="bandpass", fs=fs0, output="sos")
                zi0 = signal.sosfilt_zi(self._bp_sos)
                self._bp_zi = np.repeat(zi0[:, :, np.newaxis], c, axis=2).astype(np.float32)

    def _sample_gap(self) -> int:
        if self._scripted_queue or self.cfg.scenario.scripted_schedule:
            if self._last_scripted_word_idx == 3:
                lo, hi = self.cfg.scenario.inter_trial_gap_s
            else:
                lo, hi = self.cfg.scenario.inter_word_gap_s
            return max(1, int(self.rng.uniform(lo, hi) * self.cfg.fs_hz))
        lo, hi = self.cfg.scenario.gap_duration_s
        return max(1, int(self.rng.uniform(lo, hi) * self.cfg.fs_hz))

    def _sample_event_len(self) -> int:
        lo, hi = self.cfg.scenario.event_duration_s
        return max(1, int(self.rng.uniform(lo, hi) * self.cfg.fs_hz))

    def _init_phone_event_timeline(self, n: int, lab: str) -> None:
        seq = list(expand_word_to_phones(str(lab), self._lex))
        if not seq:
            seq = [f"@{str(lab).lower()}"]
        weights = None
        if self._phone_template_store is not None:
            weights = [self._phone_template_store.duration_weight(str(p)) for p in seq]
        self._phone_seq = seq
        self._phone_seg_lens = partition_phones_in_event(
            int(n), seq, self.rng, duration_weights=weights
        )
        self._phone_coart_env = None
        if bool(self.cfg.coarticulation_enabled) and len(seq) > 1:
            min_ov = max(
                1,
                int(round(float(self.cfg.coarticulation_min_overlap_ms) * int(self.cfg.fs_hz) / 1000.0)),
            )
            self._phone_coart_env = build_phone_coarticulation_envelopes(
                self._phone_seg_lens,
                overlap_fraction=float(self.cfg.coarticulation_overlap_fraction),
                min_overlap_samples=min_ov,
            )

    def _maybe_start_event(self) -> None:
        if self._active_label is not None:
            return
        if self._gap_remaining > 0:
            return

        if self._scripted_queue:
            item = self._scripted_queue.pop(0)
            lab = str(item.label)
            n = max(1, int(float(item.duration_s) * self.cfg.fs_hz))
            self._active_label = lab
            self._active_remaining = n
            self._current_event_start = self.sample_index
            self._event_len = int(n)
            self._pending_trial_id = int(item.trial_id)
            self._pending_word_idx = int(item.word_idx)
            self._last_scripted_word_idx = int(item.word_idx)
            if self._use_phoneme_drive:
                self._init_phone_event_timeline(int(n), str(lab))
            else:
                self._phone_seq = []
                self._phone_seg_lens = []
                self._phone_coart_env = None
            return

        if self.cfg.scenario.scripted_schedule:
            return

        if self.rng.random() > float(self.cfg.scenario.p_event):
            self._gap_remaining = self._sample_gap()
            return

        lab = _sample_choice(self.rng, self.cfg.scenario.labels, self.cfg.scenario.label_probs)
        n = self._sample_event_len()
        self._active_label = lab
        self._active_remaining = n
        self._current_event_start = self.sample_index
        self._event_len = int(n)
        if self._use_phoneme_drive:
            self._init_phone_event_timeline(int(n), str(lab))
        else:
            self._phone_seq = []
            self._phone_seg_lens = []
            self._phone_coart_env = None

    def _clock_for_sample(self, sample_index: int) -> float:
        if self.cfg.realtime_clock:
            return self._t0_wall + sample_index / float(self.cfg.fs_hz)
        return sample_index / float(self.cfg.fs_hz)

    @property
    def drive_uses_phonemes(self) -> bool:
        return bool(self._use_phoneme_drive)

    def next_chunk(self) -> FrameChunk:
        self._maybe_start_event()

        chunk_start = int(self.sample_index)
        n = int(self.chunk_samples)
        c = int(self.cfg.channels)
        fs = float(self.cfg.fs_hz)

        x = np.zeros((n, c), dtype=np.float32)
        self._sensor.apply_chunk(
            x,
            self.cfg,
            self.rng,
            chunk_start=chunk_start,
            fs=fs,
            noise_scale=float(self.cfg.noise_scale),
        )

        injected = 0
        active_label_at_start = self._active_label
        jit = float(self.cfg.spike_jitter_std_s)
        spread_motor = bool(self.cfg.muap_spatial_spread)

        if float(self.cfg.common_drive_std) > 0.0:
            phi_cd = float(np.clip(self.cfg.common_drive_ar1_phi, 0.0, 0.999))
            z = math.sqrt(max(0.0, 1.0 - phi_cd * phi_cd)) * float(self.rng.standard_normal())
            self._common_drive = phi_cd * self._common_drive + z
            rate_mod = max(0.12, 1.0 + float(self.cfg.common_drive_std) * float(self._common_drive))
        else:
            rate_mod = 1.0

        if self._motor_injector is not None:
            if self._active_label is not None and self._active_remaining > 0:
                injected = int(min(n, self._active_remaining))
            if injected > 0:
                if not self._use_phoneme_drive:
                    env = _envelope(injected, int(self.cfg.fs_hz))
                    lid = self._lid[str(self._active_label)]
                    self._motor_injector.emit_token_window(
                        x[:injected], fs, env, lid, rate_mod, jit, spread_motor
                    )
                else:
                    off = int(self._event_len - self._active_remaining)
                    if self._phone_coart_env is not None:
                        jobs = iter_coarticulated_phone_jobs(
                            off, injected, self._phone_seg_lens, self._phone_coart_env
                        )
                        for a, b, pid, env_slice in jobs:
                            phone = str(self._phone_seq[int(pid)])
                            lidp = int(self._pid[phone])
                            ph_rate = rate_mod
                            if self._phone_template_store is not None:
                                ph_rate = float(rate_mod) * float(
                                    self._phone_template_store.rate_scale(phone)
                                )
                            ar = _envelope(b - a, int(self.cfg.fs_hz))
                            env_ph = (env_slice * ar).astype(np.float32)
                            self._motor_injector.emit_token_window(
                                x[a:b], fs, env_ph, lidp, ph_rate, jit, spread_motor
                            )
                    else:
                        for a, b, pid in iter_phone_slices(off, injected, self._phone_seg_lens):
                            if b <= a:
                                continue
                            env_ph = _envelope(b - a, int(self.cfg.fs_hz))
                            phone = str(self._phone_seq[int(pid)])
                            lidp = int(self._pid[phone])
                            ph_rate = rate_mod
                            if self._phone_template_store is not None:
                                ph_rate = float(rate_mod) * float(
                                    self._phone_template_store.rate_scale(phone)
                                )
                            self._motor_injector.emit_token_window(
                                x[a:b], fs, env_ph, lidp, ph_rate, jit, spread_motor
                            )
            if injected < n:
                self._motor_injector.emit_rest_window(
                    x[injected:], fs, rate_mod, jit, spread_motor
                )
        else:
            if self._active_label is not None and self._active_remaining > 0:
                injected = int(min(n, self._active_remaining))
                prof = self.profiles[str(self._active_label)]
                pat = (np.abs(prof.weights) * float(self.cfg.token_spatial_gain)).astype(np.float32)
                env = _envelope(injected, int(self.cfg.fs_hz))
                add_muap_spikes(
                    x[:injected],
                    fs,
                    self.rng,
                    float(self.cfg.token_firing_rate_hz) * rate_mod,
                    pat,
                    self._muap,
                    float(self.cfg.muap_amplitude_uV),
                    envelope=env,
                    time_jitter_std_s=jit,
                    spread_across_channels=False,
                )

            if injected < n:
                uni = np.ones((c,), dtype=np.float32)
                add_muap_spikes(
                    x[injected:],
                    fs,
                    self.rng,
                    float(self.cfg.baseline_firing_rate_hz) * rate_mod,
                    uni,
                    self._muap,
                    float(self.cfg.muap_amplitude_uV) * float(self.cfg.baseline_amplitude_scale),
                    envelope=None,
                    time_jitter_std_s=jit,
                    spread_across_channels=False,
                )

        if not np.allclose(self._mix, np.eye(c, dtype=np.float32), atol=1e-6):
            x = (x @ self._mix.T).astype(np.float32, copy=False)

        apply_frontend_imperfections(
            x,
            self._ch_gain,
            self._ch_dc,
            adc_soft_clip_uV=self._adc_clip_uV,
        )

        if self._bp_sos is not None and self._bp_zi is not None:
            y, self._bp_zi = signal.sosfilt(self._bp_sos, x, axis=0, zi=self._bp_zi)
            x[...] = y.astype(np.float32, copy=False)

        self._sensor.apply_post_bp_lf(
            x, self.cfg, self.rng, lf_snr_scale=float(self.cfg.lf_snr_scale)
        )

        if self._active_label is not None:
            self._active_remaining -= n
            if self._active_remaining <= 0:
                end_in_chunk = n + int(self._active_remaining)
                end_in_chunk = max(0, min(n, end_in_chunk))
                start = int(self._current_event_start or chunk_start)
                end = int(chunk_start + end_in_chunk)
                word_lab = str(self._active_label)
                self.events.append(
                    SimEvent(
                        start_sample=start,
                        end_sample=end,
                        label=word_lab,
                        trial_id=self._pending_trial_id,
                        word_idx=self._pending_word_idx,
                    )
                )
                self._pending_trial_id = None
                self._pending_word_idx = None
                if self._use_phoneme_drive and self._phone_seq and self._phone_seg_lens:
                    t = int(start)
                    for p, ln in zip(self._phone_seq, self._phone_seg_lens):
                        ln_i = int(ln)
                        self.phoneme_events.append(
                            PhonemeSegment(t, t + ln_i, str(p), word_lab)
                        )
                        t += ln_i
                silence_after = n - end_in_chunk
                self._active_label = None
                self._active_remaining = 0
                self._current_event_start = None
                self._phone_seq = []
                self._phone_seg_lens = []
                self._phone_coart_env = None
                self._gap_remaining = max(0, self._sample_gap() - silence_after)
        else:
            self._gap_remaining = max(0, self._gap_remaining - n)

        seq0 = chunk_start
        t0 = float(self._clock_for_sample(seq0))
        self.sample_index += n

        return FrameChunk(
            samples=x.astype(np.float32, copy=False),
            fs_hz=int(self.cfg.fs_hz),
            t0=t0,
            seq0=seq0,
            meta={
                "sim_engine": "biophysical",
                "sim_biophysical_model": (
                    BIOPHYS_MODEL_VERSION if self._mu_lab is not None else BIOPHYS_MODEL_VERSION_LEGACY
                ),
                "sim_motor_unit_pool": self._mu_lab is not None,
                "sim_n_motor_units": int(self._mu_lab.size) if self._mu_lab is not None else 0,
                "sim_active_label": active_label_at_start,
                "sim_injected_samples": injected,
                "sim_emg_paradigm": str(self.cfg.emg_paradigm),
                "sim_token_band_hz": [float(self._resolved_token_band[0]), float(self._resolved_token_band[1])],
                "sim_literature_model": LITERATURE_MODEL_VERSION,
                "sim_noise_scale": float(self.cfg.noise_scale),
                "sim_volume_mix_mode": str(self.cfg.volume_mix_mode),
                "sim_band_limit_output": bool(self.cfg.band_limit_output),
                "sim_muap_spatial_spread": bool(spread_motor) if self._mu_lab is not None else False,
                "sim_forward_pickup": bool(self._forward_G is not None),
                "sim_n_muscle_sources": int(self.cfg.n_muscle_sources)
                if (self._mu_lab is not None and self.cfg.use_forward_pickup)
                else 0,
                "sim_conduction_delays": bool(self.cfg.use_conduction_delays) if self._mu_lab is not None else False,
                "sim_recruitment": bool(self.cfg.use_recruitment) if self._mu_lab is not None else False,
                "sim_drive_mode": str(self.cfg.scenario.drive_mode),
                "sim_realism_preset": str(self.cfg.realism_preset),
            },
        )

    def stream(self):
        while True:
            chunk = self.next_chunk()
            if self.cfg.realtime_clock:
                time.sleep(chunk.samples.shape[0] / float(self.cfg.fs_hz))
            yield chunk
