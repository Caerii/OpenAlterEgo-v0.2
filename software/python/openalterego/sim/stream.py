"""Synthetic stream generator for OpenAlterEgo.

Goal: exercise the realtime stack without hardware using **literature-aligned** *heuristics*
(see :mod:`openalterego.sim.literature` for citations and Nyquist notes).

This is *not* a biophysical muscle model. It produces:

- **Wideband-ish token bursts** whose passband follows ``emg_paradigm`` (AlterEgo-style envelope
  vs Tang/Wang-style sEMG band, always **clamped to Nyquist**).
- **White sensor noise** at configurable µV RMS (tens of µV typical in wearable EMG reports).
- **AR(1) correlated noise** (optional) as a simple stand-in for **low-frequency / motion-ish**
  correlation (cf. Tang et al. static vs motion SNR ordering).
- **Optional mains hum** (50/60 Hz) at small amplitude.
- Per-channel random walk **drift**, mild **crosstalk**, and label-specific spatial **token** patterns.

The simulator is still **stochastic and simplified**; it is meant to stress DSP/ML the way papers
describe *bands* and *rough SNR regimes*, not to reproduce recorded physiology.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Dict, Generator, Iterable, List, Literal, Optional, Tuple

import numpy as np
from scipy import signal

from ..core.types import FrameChunk
from .biophysical.sensor_pipeline import SensorNoiseState
from .literature import (
    DEFAULT_AR1_INNOVATION_SCALE,
    LITERATURE_MODEL_VERSION,
    resolve_sim_token_band,
)
from .realism import RealismPreset, apply_frontend_imperfections, realism_preset_params


@dataclass(frozen=True)
class TokenProfile:
    """Label-specific spatial pattern × band-limited noise carrier (see :mod:`openalterego.sim.literature`).

    The **passband** is taken from :func:`~openalterego.sim.literature.resolve_sim_token_band`
    (AlterEgo-style envelope vs clamped/wide sEMG literature bands, always Nyquist-safe).
    """

    label: str
    # per-channel weights (unitless), length = channels
    weights: np.ndarray
    # typical amplitude in microvolts for that label
    amplitude_uV: float = 150.0
    # base frequency content (used to shape random carrier noise)
    # Updated to match recent literature: 20-450 Hz for wide mode
    # Can be overridden for standard (1-50 Hz) or clinical (0.5-8 Hz) modes
    band_hz: Tuple[float, float] = (20.0, 450.0)


@dataclass(frozen=True)
class ScriptedWordEvent:
    """One word in a scripted trial schedule (Gowda-style)."""

    label: str
    duration_s: float
    trial_id: int = 0
    word_idx: int = 0


@dataclass
class ScenarioConfig:
    """How often tokens appear and how long they last."""

    labels: List[str] = field(default_factory=lambda: ["yes", "no", "left", "right", "select", "cancel"])
    rest_label: str = "<silence>"

    # Probability of starting an event after a gap
    p_event: float = 0.65

    # duration ranges
    event_duration_s: Tuple[float, float] = (0.35, 0.9)
    gap_duration_s: Tuple[float, float] = (0.15, 0.6)

    # Label distribution; if empty we sample uniformly from labels.
    label_probs: Optional[Dict[str, float]] = None

    # ``phoneme``: biophysical motor pool uses phone inventory + within-word timeline (see sim.phonology).
    drive_mode: Literal["word", "phoneme"] = "word"
    phone_lexicon: Optional[Dict[str, Tuple[str, ...]]] = None

    # Scripted schedule (overrides random label sampling when set).
    scripted_schedule: Optional[Tuple[ScriptedWordEvent, ...]] = None
    inter_word_gap_s: Tuple[float, float] = (0.10, 0.20)
    inter_trial_gap_s: Tuple[float, float] = (0.40, 0.80)


@dataclass
class SimStreamConfig:
    fs_hz: int = 250
    channels: int = 8
    chunk_ms: int = 40  # lower = lower latency / more overhead

    # Literature-aligned spectral paradigm (see sim/literature.py). Override bands with token_band_hz.
    emg_paradigm: str = "semg_literature_clamped"

    # signal stats (µV scale — order of magnitude consistent with wearable sEMG reports)
    noise_uV: float = 22.0
    drift_uV_per_s: float = 18.0
    crosstalk: float = 0.12  # mixes neighboring channels a bit

    # Correlated "LF / motion-ish" component: AR(1) per channel, innovation std = noise_uV * scale
    ar1_phi: float = 0.97
    ar1_innovation_scale: float = DEFAULT_AR1_INNOVATION_SCALE
    line_noise_uV: float = 0.0
    mains_freq_hz: float = 60.0

    # token shaping (burst RMS-ish scale; literature: tens–hundreds of µV depending on site/gain)
    token_amplitude_uV: float = 185.0
    token_snr_boost: float = 1.0  # >1 makes tokens clearer
    # If None, band comes from emg_paradigm (+ Nyquist). Set explicitly to override.
    token_band_hz: Optional[Tuple[float, float]] = None
    seed: int = 1337

    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)

    # If True, t0 uses time.time() (wall clock). If False, uses a monotonic synthetic clock.
    realtime_clock: bool = True
    # Extra LF/motion/mains-harmonic morphology + per-channel gain/DC (default off for fast tests).
    realism_preset: RealismPreset = "off"


@dataclass
class SimEvent:
    start_sample: int
    end_sample: int
    label: str
    trial_id: Optional[int] = None
    word_idx: Optional[int] = None


def _stable_label_seed(seed: int, label: str) -> int:
    """Create a stable seed from (seed,label) (no dependence on Python's randomized hash())."""
    h = 2166136261
    for b in label.encode("utf-8"):
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return (seed ^ h) & 0xFFFFFFFF


def _make_profiles(
    labels: Iterable[str], 
    channels: int, 
    *, 
    seed: int, 
    amplitude_uV: float,
    band_hz: Tuple[float, float] = (20.0, 450.0),
) -> Dict[str, TokenProfile]:
    """Create token profiles with realistic frequency bands.
    
    Parameters
    ----------
    band_hz:
        Frequency band for token generation (default: 20-450 Hz for modern EMG)
    """
    profiles: Dict[str, TokenProfile] = {}
    for lab in labels:
        rng = np.random.default_rng(_stable_label_seed(seed, lab))
        w = rng.normal(size=(channels,)).astype(np.float32)
        w /= (np.linalg.norm(w) + 1e-8)
        # bias toward a couple "strong" channels so classes separate
        idx = rng.choice(channels, size=max(1, channels // 4), replace=False)
        w[idx] *= 1.6
        w /= (np.linalg.norm(w) + 1e-8)
        profiles[lab] = TokenProfile(
            label=lab, 
            weights=w, 
            amplitude_uV=float(amplitude_uV),
            band_hz=band_hz,
        )
    return profiles


def _sample_choice(rng: np.random.Generator, labels: List[str], probs: Optional[Dict[str, float]]) -> str:
    if not probs:
        return str(rng.choice(labels))
    p = np.array([float(probs.get(l, 0.0)) for l in labels], dtype=np.float64)
    if np.sum(p) <= 0:
        return str(rng.choice(labels))
    p = p / np.sum(p)
    return str(rng.choice(labels, p=p))


def _bandpass_carrier(rng: np.random.Generator, n: int, *, fs_hz: int, band: Tuple[float, float]) -> np.ndarray:
    """Generate band-limited noise carrier (n,) with approximate passband."""
    sos = signal.butter(3, list(band), btype="bandpass", fs=fs_hz, output="sos")
    white = rng.standard_normal(size=(n,)).astype(np.float32)
    y = signal.sosfilt(sos, white).astype(np.float32)
    y = y / (np.std(y) + 1e-6)
    return y


def _envelope(n: int, fs_hz: int) -> np.ndarray:
    """Smooth attack/decay envelope."""
    attack = min(n, max(1, int(0.06 * fs_hz)))
    release = min(n, max(1, int(0.10 * fs_hz)))
    env = np.ones((n,), dtype=np.float32)
    env[:attack] = np.linspace(0.0, 1.0, attack, dtype=np.float32)
    env[-release:] = np.linspace(1.0, 0.0, release, dtype=np.float32)
    if n >= 8:
        win = signal.windows.hann(min(n, 64), sym=False).astype(np.float32)
        win /= np.sum(win)
        env = np.convolve(env, win, mode="same").astype(np.float32)
        env = np.clip(env, 0.0, 1.0)
    return env


class SimStream:
    """Stateful, chunked simulator."""

    def __init__(self, config: SimStreamConfig):
        self.cfg = config
        self.rng = np.random.default_rng(int(config.seed))
        self.chunk_samples = max(1, int(config.fs_hz * config.chunk_ms / 1000))

        token_band = resolve_sim_token_band(
            int(config.fs_hz),
            str(config.emg_paradigm),
            config.token_band_hz,
        )
        self._resolved_token_band: Tuple[float, float] = token_band

        self.profiles = _make_profiles(
            config.scenario.labels,
            config.channels,
            seed=int(config.seed),
            amplitude_uV=float(config.token_amplitude_uV),
            band_hz=token_band,
        )

        # timeline state
        self.sample_index = 0
        self._t0_wall = time.time()
        self._active_label: Optional[str] = None
        self._active_remaining = 0
        self._gap_remaining = self._sample_gap()
        self._current_event_start: Optional[int] = None

        self.events: List[SimEvent] = []

        self._sensor = SensorNoiseState(int(config.channels), self.rng)
        hrp = realism_preset_params(str(config.realism_preset))
        c = int(config.channels)
        if hrp.channel_gain_log_sigma > 0.0:
            self._ch_gain = self.rng.lognormal(
                0.0, float(hrp.channel_gain_log_sigma), size=(c,)
            ).astype(np.float32)
        else:
            self._ch_gain = np.ones((c,), dtype=np.float32)
        if hrp.channel_dc_uV_half_range > 0.0:
            half = float(hrp.channel_dc_uV_half_range)
            self._ch_dc = self.rng.uniform(-half, half, size=(c,)).astype(np.float32)
        else:
            self._ch_dc = np.zeros((c,), dtype=np.float32)
        self._adc_clip_uV: Optional[float] = (
            float(hrp.adc_soft_clip_uV) if hrp.adc_soft_clip_uV is not None else None
        )

        # channel mixing matrix for mild cross-talk
        self._mix = np.eye(config.channels, dtype=np.float32)
        if config.crosstalk > 0:
            ct = float(config.crosstalk)
            for i in range(config.channels):
                if i - 1 >= 0:
                    self._mix[i, i - 1] += ct
                if i + 1 < config.channels:
                    self._mix[i, i + 1] += ct
            self._mix = (self._mix / np.sum(self._mix, axis=1, keepdims=True)).astype(np.float32)

    def _sample_gap(self) -> int:
        lo, hi = self.cfg.scenario.gap_duration_s
        return max(1, int(self.rng.uniform(lo, hi) * self.cfg.fs_hz))

    def _sample_event_len(self) -> int:
        lo, hi = self.cfg.scenario.event_duration_s
        return max(1, int(self.rng.uniform(lo, hi) * self.cfg.fs_hz))

    def _maybe_start_event(self) -> None:
        if self._active_label is not None:
            return
        if self._gap_remaining > 0:
            return
        if self.rng.random() > float(self.cfg.scenario.p_event):
            self._gap_remaining = self._sample_gap()
            return

        lab = _sample_choice(self.rng, self.cfg.scenario.labels, self.cfg.scenario.label_probs)
        n = self._sample_event_len()
        self._active_label = lab
        self._active_remaining = n
        self._current_event_start = self.sample_index

    def _clock_for_sample(self, sample_index: int) -> float:
        if self.cfg.realtime_clock:
            return self._t0_wall + sample_index / float(self.cfg.fs_hz)
        return sample_index / float(self.cfg.fs_hz)

    def next_chunk(self) -> FrameChunk:
        """Generate the next chunk."""
        self._maybe_start_event()

        chunk_start = int(self.sample_index)
        n = int(self.chunk_samples)
        c = int(self.cfg.channels)

        x = np.zeros((n, c), dtype=np.float32)
        shim = SimpleNamespace(
            electrode_noise_uV=float(self.cfg.noise_uV),
            drift_uV_per_s=float(self.cfg.drift_uV_per_s),
            ar1_phi=float(self.cfg.ar1_phi),
            ar1_innovation_scale=float(self.cfg.ar1_innovation_scale),
            line_noise_uV=float(self.cfg.line_noise_uV),
            mains_freq_hz=float(self.cfg.mains_freq_hz),
            realism_preset=str(self.cfg.realism_preset),
        )
        self._sensor.apply_chunk(
            x,
            shim,
            self.rng,
            chunk_start=int(chunk_start),
            fs=float(self.cfg.fs_hz),
            noise_scale=1.0,
        )

        # event injection into the beginning of the chunk
        injected = 0
        active_label_at_start = self._active_label
        if self._active_label is not None and self._active_remaining > 0:
            injected = int(min(n, self._active_remaining))
            prof = self.profiles[str(self._active_label)]
            env = _envelope(injected, self.cfg.fs_hz)[:, None]  # (k,1)
            carrier = _bandpass_carrier(self.rng, injected, fs_hz=self.cfg.fs_hz, band=prof.band_hz)[:, None]
            pat = prof.weights[None, :].astype(np.float32)
            amp = float(prof.amplitude_uV) * float(self.cfg.token_snr_boost)
            x[:injected, :] += amp * env * carrier * pat

        # apply mild cross-talk mixing
        if self.cfg.crosstalk > 0:
            x = x @ self._mix.T

        apply_frontend_imperfections(
            x,
            self._ch_gain,
            self._ch_dc,
            adc_soft_clip_uV=self._adc_clip_uV,
        )

        # update scenario counters
        if self._active_label is not None:
            self._active_remaining -= n
            if self._active_remaining <= 0:
                # event ended inside this chunk
                end_in_chunk = n + int(self._active_remaining)  # (0..n)
                end_in_chunk = max(0, min(n, end_in_chunk))
                start = int(self._current_event_start or chunk_start)
                end = int(chunk_start + end_in_chunk)
                self.events.append(SimEvent(start_sample=start, end_sample=end, label=str(self._active_label)))

                # start a new gap, but account for the already-simulated silence remainder
                silence_after = n - end_in_chunk
                self._active_label = None
                self._active_remaining = 0
                self._current_event_start = None
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
                "sim_active_label": active_label_at_start,
                "sim_injected_samples": injected,
                "sim_emg_paradigm": str(self.cfg.emg_paradigm),
                "sim_token_band_hz": [float(self._resolved_token_band[0]), float(self._resolved_token_band[1])],
                "sim_literature_model": LITERATURE_MODEL_VERSION,
                "sim_realism_preset": str(self.cfg.realism_preset),
            },
        )

    def stream(self) -> Generator[FrameChunk, None, None]:
        """Infinite generator."""
        while True:
            chunk = self.next_chunk()
            if self.cfg.realtime_clock:
                time.sleep(chunk.samples.shape[0] / float(self.cfg.fs_hz))
            yield chunk
