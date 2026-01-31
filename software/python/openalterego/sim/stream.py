"""Synthetic stream generator for OpenAlterEgo.

Goal: let you exercise the whole realtime stack without touching hardware.

This is *not* a biophysical muscle model. It's a pragmatic generator that produces:
- multi-channel correlated noise,
- low-frequency drift/motion-ish components,
- token-shaped activation bursts with label-specific spatial patterns.

The point is not realism — it's *controllable complexity* so your pipeline doesn't fall apart
the moment you swap simulated data for real data.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, Generator, Iterable, List, Optional, Tuple

import numpy as np
from scipy import signal

from ..core.types import FrameChunk


@dataclass(frozen=True)
class TokenProfile:
    """Defines a label-specific spatial + temporal signature."""

    label: str
    # per-channel weights (unitless), length = channels
    weights: np.ndarray
    # typical amplitude in microvolts for that label
    amplitude_uV: float = 150.0
    # base frequency content (used to shape random carrier noise)
    band_hz: Tuple[float, float] = (2.0, 45.0)


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


@dataclass
class SimStreamConfig:
    fs_hz: int = 250
    channels: int = 8
    chunk_ms: int = 40  # lower = lower latency / more overhead

    # signal stats
    noise_uV: float = 18.0
    drift_uV_per_s: float = 15.0
    crosstalk: float = 0.10  # mixes neighboring channels a bit

    # token shaping
    token_amplitude_uV: float = 170.0
    token_snr_boost: float = 1.0  # >1 makes tokens clearer
    seed: int = 1337

    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)

    # If True, t0 uses time.time() (wall clock). If False, uses a monotonic synthetic clock.
    realtime_clock: bool = True


@dataclass
class SimEvent:
    start_sample: int
    end_sample: int
    label: str


def _stable_label_seed(seed: int, label: str) -> int:
    """Create a stable seed from (seed,label) (no dependence on Python's randomized hash())."""
    h = 2166136261
    for b in label.encode("utf-8"):
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return (seed ^ h) & 0xFFFFFFFF


def _make_profiles(labels: Iterable[str], channels: int, *, seed: int, amplitude_uV: float) -> Dict[str, TokenProfile]:
    profiles: Dict[str, TokenProfile] = {}
    for lab in labels:
        rng = np.random.default_rng(_stable_label_seed(seed, lab))
        w = rng.normal(size=(channels,)).astype(np.float32)
        w /= (np.linalg.norm(w) + 1e-8)
        # bias toward a couple "strong" channels so classes separate
        idx = rng.choice(channels, size=max(1, channels // 4), replace=False)
        w[idx] *= 1.6
        w /= (np.linalg.norm(w) + 1e-8)
        profiles[lab] = TokenProfile(label=lab, weights=w, amplitude_uV=float(amplitude_uV))
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
        self.profiles = _make_profiles(
            config.scenario.labels,
            config.channels,
            seed=int(config.seed),
            amplitude_uV=float(config.token_amplitude_uV),
        )

        # timeline state
        self.sample_index = 0
        self._t0_wall = time.time()
        self._active_label: Optional[str] = None
        self._active_remaining = 0
        self._gap_remaining = self._sample_gap()
        self._current_event_start: Optional[int] = None

        self.events: List[SimEvent] = []

        # drift state (random walk)
        self._drift = np.zeros((config.channels,), dtype=np.float32)

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

        # baseline noise
        x = self.rng.standard_normal(size=(n, self.cfg.channels)).astype(np.float32) * float(self.cfg.noise_uV)

        # drift: random walk (per second scale)
        drift_step = float(self.cfg.drift_uV_per_s) / math.sqrt(float(self.cfg.fs_hz))
        self._drift += self.rng.normal(scale=drift_step, size=(self.cfg.channels,)).astype(np.float32)
        x += self._drift[None, :]

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
            meta={"sim_active_label": active_label_at_start, "sim_injected_samples": injected},
        )

    def stream(self) -> Generator[FrameChunk, None, None]:
        """Infinite generator."""
        while True:
            chunk = self.next_chunk()
            if self.cfg.realtime_clock:
                time.sleep(chunk.samples.shape[0] / float(self.cfg.fs_hz))
            yield chunk
