"""Benchmark biophysical synthesis: scaling laws, chunk tuning, throughput targets."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .stream import BiophysicalSimStream, BiophysicalSimStreamConfig
from ..stream import ScenarioConfig


@dataclass(frozen=True)
class ChunkTiming:
    fs_hz: int
    channels: int
    n_motor_units: int
    chunk_ms: int
    synth_mode: str
    n_chunks: int
    wall_s: float
    samples_per_s: float
    ms_per_chunk: float
    realtime_factor: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fs_hz": self.fs_hz,
            "channels": self.channels,
            "n_motor_units": self.n_motor_units,
            "chunk_ms": self.chunk_ms,
            "synth_mode": self.synth_mode,
            "n_chunks": self.n_chunks,
            "wall_s": round(self.wall_s, 4),
            "samples_per_s": round(self.samples_per_s, 1),
            "ms_per_chunk": round(self.ms_per_chunk, 3),
            "realtime_factor": round(self.realtime_factor, 1),
        }


@dataclass
class ScalingReport:
    """Empirical scaling + recommended operating point."""

    timings: List[ChunkTiming] = field(default_factory=list)
    recommended_chunk_ms: int = 40
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timings": [t.to_dict() for t in self.timings],
            "recommended_chunk_ms": self.recommended_chunk_ms,
            "scaling_laws": SCALING_LAWS_TEXT,
            "notes": list(self.notes),
        }


SCALING_LAWS_TEXT = """
Dominant cost model (biophysical v5/v6, synth_mode=fast|numba|rust):
  T_chunk ≈ α·(n/fs)·n_mu·log(n) + β·n·c + γ·c·n_sections
  n = fs·chunk_ms/1000 samples/chunk
  n_spikes ≈ Σ_i rate_i · (n/fs)  (Poisson; token windows ↑ rates)

Backend ladder (auto): numba > rust > python convolve/scatter

Empirical scaling (hold chunk_ms=40ms):
  fs:        ~O(fs^0.3–0.9); use --auto-chunk above 1 kHz
  channels:  ~O(c^0.7–1.0) motor spread + bandpass
  n_mu:      ~O(n_mu) scatter/convolve per unit

Realtime factor = (chunk_ms) / (ms_per_chunk). Target ≥ 20× for dataset gen.
"""


def _make_stream(
    *,
    fs_hz: int,
    channels: int,
    n_motor_units: int,
    chunk_ms: int,
    synth_mode: str,
    seed: int,
    realism: str = "tang",
    use_conduction_delays: bool = False,
) -> BiophysicalSimStream:
    sc = ScenarioConfig(labels=["yes", "no", "left", "right"], p_event=0.75)
    cfg = BiophysicalSimStreamConfig(
        fs_hz=int(fs_hz),
        channels=int(channels),
        n_motor_units=int(n_motor_units),
        chunk_ms=int(chunk_ms),
        seed=int(seed),
        scenario=sc,
        realtime_clock=False,
        realism_preset=realism,  # type: ignore[arg-type]
        use_conduction_delays=bool(use_conduction_delays),
        synth_mode=synth_mode,  # type: ignore[arg-type]
        use_batch_synthesis=synth_mode != "legacy",
    )
    return BiophysicalSimStream(cfg)


def benchmark_chunk(
    *,
    fs_hz: int = 500,
    channels: int = 8,
    n_motor_units: int = 48,
    chunk_ms: int = 40,
    synth_mode: str = "fast",
    n_chunks: int = 120,
    seed: int = 0,
    warmup: int = 8,
    use_conduction_delays: bool = False,
) -> ChunkTiming:
    sim = _make_stream(
        fs_hz=fs_hz,
        channels=channels,
        n_motor_units=n_motor_units,
        chunk_ms=chunk_ms,
        synth_mode=synth_mode,
        seed=seed,
        use_conduction_delays=use_conduction_delays,
    )
    for _ in range(warmup):
        sim.next_chunk()
    t0 = time.perf_counter()
    for _ in range(n_chunks):
        sim.next_chunk()
    wall = time.perf_counter() - t0
    n_samp = n_chunks * int(fs_hz * chunk_ms / 1000) * int(channels)
    sps = float(n_samp) / max(wall, 1e-9)
    ms_pc = 1000.0 * wall / float(n_chunks)
    rt = float(chunk_ms) / max(ms_pc, 1e-6)
    return ChunkTiming(
        fs_hz=int(fs_hz),
        channels=int(channels),
        n_motor_units=int(n_motor_units),
        chunk_ms=int(chunk_ms),
        synth_mode=str(synth_mode),
        n_chunks=int(n_chunks),
        wall_s=float(wall),
        samples_per_s=float(sps),
        ms_per_chunk=float(ms_pc),
        realtime_factor=float(rt),
    )


def recommend_chunk_ms(
    fs_hz: int,
    *,
    target_realtime_factor: float = 25.0,
    channels: int = 8,
    n_motor_units: int = 48,
    synth_mode: str = "fast",
    candidates: Optional[List[int]] = None,
) -> Tuple[int, ChunkTiming]:
    """Pick chunk_ms maximizing throughput while staying ≥ target realtime factor."""
    if candidates is None:
        # Larger chunks amortize Python overhead; cap for streaming latency.
        base = max(20, int(1000 * 10 / max(fs_hz, 100)))  # ~10ms min at high fs
        candidates = sorted(
            {
                max(20, base),
                40,
                60,
                80,
                100,
                120,
                160,
                200,
                240,
                320,
            }
        )
    best_ms = int(candidates[0])
    best_t: Optional[ChunkTiming] = None
    for ms in candidates:
        t = benchmark_chunk(
            fs_hz=fs_hz,
            channels=channels,
            n_motor_units=n_motor_units,
            chunk_ms=ms,
            synth_mode=synth_mode,
            n_chunks=60,
        )
        if t.realtime_factor >= target_realtime_factor:
            if best_t is None or t.ms_per_chunk < best_t.ms_per_chunk:
                best_ms = ms
                best_t = t
    if best_t is None:
        best_t = benchmark_chunk(
            fs_hz=fs_hz,
            channels=channels,
            n_motor_units=n_motor_units,
            chunk_ms=best_ms,
            synth_mode=synth_mode,
            n_chunks=60,
        )
    return best_ms, best_t


def run_scaling_sweep(
    *,
    fs_values: Optional[List[int]] = None,
    channel_values: Optional[List[int]] = None,
    mu_values: Optional[List[int]] = None,
    chunk_ms: int = 40,
    synth_mode: str = "fast",
) -> ScalingReport:
    fs_values = fs_values or [250, 500, 1000]
    channel_values = channel_values or [4, 8, 16]
    mu_values = mu_values or [24, 48, 96]
    report = ScalingReport()
    seed = 0
    for fs in fs_values:
        t = benchmark_chunk(fs_hz=fs, channels=8, n_motor_units=48, chunk_ms=chunk_ms, synth_mode=synth_mode, seed=seed)
        report.timings.append(t)
        seed += 1
    for ch in channel_values:
        t = benchmark_chunk(fs_hz=500, channels=ch, n_motor_units=48, chunk_ms=chunk_ms, synth_mode=synth_mode, seed=seed)
        report.timings.append(t)
        seed += 1
    for n_mu in mu_values:
        t = benchmark_chunk(fs_hz=500, channels=8, n_motor_units=n_mu, chunk_ms=chunk_ms, synth_mode=synth_mode, seed=seed)
        report.timings.append(t)
        seed += 1

    rec_ms, rec_t = recommend_chunk_ms(1000, channels=8, n_motor_units=48, synth_mode=synth_mode)
    report.recommended_chunk_ms = rec_ms
    report.notes.append(
        f"At 1000 Hz, chunk_ms={rec_ms} -> {rec_t.realtime_factor:.0f}x realtime "
        f"({rec_t.samples_per_s/1e6:.2f} Msamp/s)."
    )
    t_fast = benchmark_chunk(fs_hz=500, channels=8, n_motor_units=48, chunk_ms=40, synth_mode="fast", seed=99)
    t_batch = benchmark_chunk(fs_hz=500, channels=8, n_motor_units=48, chunk_ms=40, synth_mode="batch", seed=99)
    speedup = t_batch.ms_per_chunk / max(t_fast.ms_per_chunk, 1e-6)
    report.notes.append(f"fast vs batch speedup @ 500Hz/8ch/48MU: {speedup:.1f}x")
    return report


def run_extended_scaling_sweep(
    *,
    chunk_ms: int = 40,
    synth_mode: str = "fast",
    use_conduction_delays: bool = False,
) -> ScalingReport:
    """High-fs / large-scale sweep: up to 4 kHz, 192 MU, 32 channels."""
    from .accel_backend import active_backend_label

    fs_values = [500, 1000, 1500, 2000, 4000]
    channel_values = [8, 16, 32]
    mu_values = [48, 96, 192]
    report = ScalingReport()
    seed = 0
    for fs in fs_values:
        ms = chunk_ms if fs <= 1000 else max(chunk_ms, auto_chunk_ms_for_fs(fs))
        t = benchmark_chunk(
            fs_hz=fs, channels=8, n_motor_units=48, chunk_ms=ms,
            synth_mode=synth_mode, seed=seed, use_conduction_delays=use_conduction_delays,
        )
        report.timings.append(t)
        seed += 1
    for ch in channel_values:
        t = benchmark_chunk(
            fs_hz=2000, channels=ch, n_motor_units=48, chunk_ms=80,
            synth_mode=synth_mode, seed=seed, use_conduction_delays=use_conduction_delays,
        )
        report.timings.append(t)
        seed += 1
    for n_mu in mu_values:
        t = benchmark_chunk(
            fs_hz=2000, channels=8, n_motor_units=n_mu, chunk_ms=80,
            synth_mode=synth_mode, seed=seed, use_conduction_delays=use_conduction_delays,
        )
        report.timings.append(t)
        seed += 1

    for mode in ("fast", "numba", "rust", "batch"):
        try:
            t = benchmark_chunk(
                fs_hz=2000, channels=8, n_motor_units=96, chunk_ms=80,
                synth_mode=mode, seed=42, use_conduction_delays=use_conduction_delays,
            )
            report.notes.append(
                f"mode={mode} @ 2kHz/8ch/96MU/80ms -> {t.realtime_factor:.0f}x RT "
                f"({t.ms_per_chunk:.2f} ms/chunk)"
            )
        except RuntimeError as exc:
            report.notes.append(f"mode={mode} skipped: {exc}")

    rec_ms, rec_t = recommend_chunk_ms(4000, channels=8, n_motor_units=48, synth_mode=synth_mode)
    report.recommended_chunk_ms = rec_ms
    report.notes.append(
        f"backend={active_backend_label()} | 4 kHz chunk_ms={rec_ms} -> "
        f"{rec_t.realtime_factor:.0f}x RT ({rec_t.samples_per_s/1e6:.2f} Msamp/s)"
    )
    return report


def auto_chunk_ms_for_fs(fs_hz: int, channels: int = 8) -> int:
    """Heuristic chunk size for dataset generation (not streaming serve)."""
    ms, _ = recommend_chunk_ms(int(fs_hz), channels=int(channels))
    return int(ms)
