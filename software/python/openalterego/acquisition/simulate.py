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
from typing import Generator, List

import numpy as np

from ..core.types import FrameChunk
from ..sim.stream import ScenarioConfig, SimStream, SimStreamConfig


@dataclass
class SimConfig:
    fs_hz: int = 250
    channels: int = 8
    chunk_ms: int = 40

    labels: List[str] = None  # filled in __post_init__
    seed: int = 1337
    noise_uV: float = 18.0
    drift_uV_per_s: float = 15.0
    crosstalk: float = 0.10
    p_event: float = 0.65

    realtime_clock: bool = True

    def __post_init__(self) -> None:
        if self.labels is None:
            self.labels = ["yes", "no", "left", "right", "select", "cancel"]


def stream_simulated_chunks(cfg: SimConfig) -> Generator[FrameChunk, None, None]:
    sc = ScenarioConfig(labels=list(cfg.labels), p_event=float(cfg.p_event))
    sim_cfg = SimStreamConfig(
        fs_hz=int(cfg.fs_hz),
        channels=int(cfg.channels),
        chunk_ms=int(cfg.chunk_ms),
        noise_uV=float(cfg.noise_uV),
        drift_uV_per_s=float(cfg.drift_uV_per_s),
        crosstalk=float(cfg.crosstalk),
        seed=int(cfg.seed),
        scenario=sc,
        realtime_clock=bool(cfg.realtime_clock),
    )
    sim = SimStream(sim_cfg)
    yield from sim.stream()


def stream_simulated(cfg: SimConfig) -> Generator[np.ndarray, None, None]:
    """Legacy helper that yields just the sample arrays."""
    for chunk in stream_simulated_chunks(cfg):
        yield chunk.samples
