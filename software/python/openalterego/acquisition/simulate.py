"""Simulated acquisition sources.

This wraps :class:`openalterego.sim.stream.SimStream` so you can test the realtime pipeline
with *no hardware*.

Two helpers are provided:
- :func:`stream_simulated_chunks` yields :class:`~openalterego.core.types.FrameChunk` objects
- :func:`stream_simulated` yields raw numpy arrays (legacy convenience)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Generator, List, Literal, Optional, Tuple, cast

import numpy as np

from ..core.types import FrameChunk
from ..sim.biophysical.stream import BiophysicalSimStream, BiophysicalSimStreamConfig
from ..sim.literature import DEFAULT_AR1_INNOVATION_SCALE
from ..sim.stream import ScenarioConfig, SimStream, SimStreamConfig


@dataclass
class SimConfig:
    fs_hz: int = 250
    channels: int = 8
    chunk_ms: int = 40

    labels: List[str] = None  # filled in __post_init__
    seed: int = 1337
    emg_paradigm: str = "semg_literature_clamped"
    noise_uV: float = 22.0
    drift_uV_per_s: float = 18.0
    crosstalk: float = 0.12
    p_event: float = 0.65
    ar1_phi: float = 0.97
    ar1_innovation_scale: float = DEFAULT_AR1_INNOVATION_SCALE
    line_noise_uV: float = 0.0
    mains_freq_hz: float = 60.0
    token_band_hz: Optional[Tuple[float, float]] = None
    token_amplitude_uV: float = 185.0

    realtime_clock: bool = True
    sim_engine: str = "heuristic"
    # ``off`` / ``wearable`` / ``tang`` / ``field`` (see ``sim.realism``). Empty = use engine defaults.
    realism_preset: str = ""
    noise_scale: float = 1.0
    montage_name: str = ""
    scenario: Optional[ScenarioConfig] = None

    def __post_init__(self) -> None:
        if self.labels is None:
            self.labels = ["yes", "no", "left", "right", "select", "cancel"]


def open_sim_from_config(cfg: SimConfig) -> SimStream | BiophysicalSimStream:
    """Construct a simulator instance (for collection / event ground truth)."""
    sc = cfg.scenario if cfg.scenario is not None else ScenarioConfig(labels=list(cfg.labels), p_event=float(cfg.p_event))
    rp = str(cfg.realism_preset).strip().lower()
    valid_rp = ("off", "wearable", "tang", "field")
    if rp and rp not in valid_rp:
        raise ValueError(
            f"SimConfig.realism_preset must be one of {valid_rp} or '', got {cfg.realism_preset!r}"
        )
    sim_rp = cast(Literal["off", "wearable", "tang", "field"], rp) if rp else None
    montage = str(cfg.montage_name).strip() or None

    if str(cfg.sim_engine).lower() == "biophysical":
        bkwargs = dict(
            fs_hz=int(cfg.fs_hz),
            channels=int(cfg.channels),
            chunk_ms=int(cfg.chunk_ms),
            seed=int(cfg.seed),
            scenario=sc,
            realtime_clock=bool(cfg.realtime_clock),
            emg_paradigm=str(cfg.emg_paradigm),
            token_band_hz=cfg.token_band_hz,
            electrode_noise_uV=float(cfg.noise_uV),
            drift_uV_per_s=float(cfg.drift_uV_per_s),
            crosstalk=float(cfg.crosstalk),
            ar1_phi=float(cfg.ar1_phi),
            ar1_innovation_scale=float(cfg.ar1_innovation_scale),
            line_noise_uV=float(cfg.line_noise_uV),
            mains_freq_hz=float(cfg.mains_freq_hz),
            noise_scale=float(cfg.noise_scale),
            montage_name=montage,
        )
        if sim_rp is not None:
            bkwargs["realism_preset"] = sim_rp
        return BiophysicalSimStream(BiophysicalSimStreamConfig(**bkwargs))

    sim_kw = dict(
        fs_hz=int(cfg.fs_hz),
        channels=int(cfg.channels),
        chunk_ms=int(cfg.chunk_ms),
        emg_paradigm=str(cfg.emg_paradigm),
        noise_uV=float(cfg.noise_uV),
        drift_uV_per_s=float(cfg.drift_uV_per_s),
        crosstalk=float(cfg.crosstalk),
        ar1_phi=float(cfg.ar1_phi),
        ar1_innovation_scale=float(cfg.ar1_innovation_scale),
        line_noise_uV=float(cfg.line_noise_uV),
        mains_freq_hz=float(cfg.mains_freq_hz),
        token_band_hz=cfg.token_band_hz,
        token_amplitude_uV=float(cfg.token_amplitude_uV),
        seed=int(cfg.seed),
        scenario=sc,
        realtime_clock=bool(cfg.realtime_clock),
    )
    if sim_rp is not None:
        sim_kw["realism_preset"] = sim_rp
    return SimStream(SimStreamConfig(**sim_kw))


def stream_simulated_chunks(cfg: SimConfig) -> Generator[FrameChunk, None, None]:
    sim = open_sim_from_config(cfg)
    yield from sim.stream()


def stream_simulated(cfg: SimConfig) -> Generator[np.ndarray, None, None]:
    """Legacy helper that yields just the sample arrays."""
    for chunk in stream_simulated_chunks(cfg):
        yield chunk.samples
