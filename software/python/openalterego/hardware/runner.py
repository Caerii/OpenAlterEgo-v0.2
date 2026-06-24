"""Run hardware DSL: simulate chunks, virtual BLE packet stream, quality smoke test."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import AsyncIterator, Dict, List, Optional

import numpy as np

from ..acquisition.packet import parse_oa_v1
from ..acquisition.simulate import stream_simulated_chunks
from ..acquisition.virtual import VirtualBleSpec, virtual_notifications
from ..core.types import FrameChunk
from ..dsp.quality import assess_signal_quality
from ..dsp.emg_config import emg_signal_band_hz_for_quality
from ..sim.transport import LinkConfig, apply_link
from .resolve import ResolvedHardware, resolve_all, resolve_sim_config, resolve_virtual_ble_spec
from .schema import HardwareSpec


@dataclass
class SimulateReport:
    """Summary from a short hardware-bound simulation run."""

    n_chunks: int
    n_samples: int
    duration_s: float
    mean_snr_db: Optional[float]
    mean_motion_index: Optional[float]
    packets_parsed: int = 0
    packets_lost: int = 0
    issues: List[str] = None

    def __post_init__(self) -> None:
        if self.issues is None:
            self.issues = []


def run_chunk_simulation(
    spec: HardwareSpec,
    *,
    duration_s: float = 5.0,
    max_chunks: int = 500,
) -> tuple[List[FrameChunk], SimulateReport]:
    """Stream ``FrameChunk``s using resolved :class:`SimConfig``."""
    cfg = resolve_sim_config(spec)
    cfg.realtime_clock = False
    resolved = resolve_all(spec)
    chunks: List[FrameChunk] = []
    n_samples = 0
    snrs: List[float] = []
    motions: List[float] = []
    band = emg_signal_band_hz_for_quality(resolved.preprocess_mode, float(cfg.fs_hz))

    for ch in stream_simulated_chunks(cfg):
        chunks.append(ch)
        n_samples += int(ch.samples.shape[0])
        if ch.samples.size:
            q = assess_signal_quality(
                ch.samples,
                fs_hz=float(cfg.fs_hz),
                signal_band_hz=band,
            )
            if q.snr_db is not None:
                snrs.append(float(q.snr_db))
            if q.motion_index is not None:
                motions.append(float(q.motion_index))
        if n_samples >= duration_s * cfg.fs_hz or len(chunks) >= max_chunks:
            break

    report = SimulateReport(
        n_chunks=len(chunks),
        n_samples=n_samples,
        duration_s=n_samples / float(cfg.fs_hz) if cfg.fs_hz else 0.0,
        mean_snr_db=float(np.mean(snrs)) if snrs else None,
        mean_motion_index=float(np.mean(motions)) if motions else None,
    )
    return chunks, report


async def _virtual_ble_packets(
    vspec: VirtualBleSpec,
    *,
    duration_s: float,
) -> AsyncIterator[bytes]:
    target_samples = int(vspec.fs_hz * duration_s)
    n = 0
    async for pkt in apply_link(virtual_notifications(vspec), cfg=vspec.link):
        yield pkt
        if vspec.packet_format == "oa_v1":
            frames, info = parse_oa_v1(pkt, afe=vspec.afe)
            n += int(frames.shape[0])
        else:
            n += len(pkt) // (2 * vspec.channels)
        if n >= target_samples:
            break


async def run_virtual_ble_simulation(
    spec: HardwareSpec,
    *,
    duration_s: float = 3.0,
) -> SimulateReport:
    """Exercise OA v1 pack → link impairments → parse at byte level."""
    vspec = resolve_virtual_ble_spec(spec)
    resolved = resolve_all(spec)
    band = emg_signal_band_hz_for_quality(
        resolved.preprocess_mode, float(vspec.fs_hz)
    )
    snrs: List[float] = []
    motions: List[float] = []
    parsed = 0
    lost = 0
    n_samples = 0
    expected_seq: Optional[int] = None

    async for pkt in _virtual_ble_packets(vspec, duration_s=duration_s):
        if vspec.packet_format != "oa_v1":
            parsed += 1
            continue
        try:
            frames, info = parse_oa_v1(pkt, afe=vspec.afe)
            parsed += 1
            idx = int(info["sample_index0"])
            if expected_seq is not None and idx > expected_seq:
                lost += 1
            expected_seq = idx + int(frames.shape[0])
            n_samples += int(frames.shape[0])
            q = assess_signal_quality(frames, fs_hz=float(vspec.fs_hz), signal_band_hz=band)
            if q.snr_db is not None:
                snrs.append(float(q.snr_db))
            if q.motion_index is not None:
                motions.append(float(q.motion_index))
        except ValueError:
            lost += 1

    return SimulateReport(
        n_chunks=0,
        n_samples=n_samples,
        duration_s=n_samples / float(vspec.fs_hz) if vspec.fs_hz else 0.0,
        mean_snr_db=float(np.mean(snrs)) if snrs else None,
        mean_motion_index=float(np.mean(motions)) if motions else None,
        packets_parsed=parsed,
        packets_lost=lost,
    )


def run_virtual_ble_simulation_sync(spec: HardwareSpec, *, duration_s: float = 3.0) -> SimulateReport:
    return asyncio.run(run_virtual_ble_simulation(spec, duration_s=duration_s))


def simulate_report_dict(report: SimulateReport) -> Dict[str, object]:
    return {
        "n_chunks": report.n_chunks,
        "n_samples": report.n_samples,
        "duration_s": round(report.duration_s, 4),
        "mean_snr_db": None if report.mean_snr_db is None else round(report.mean_snr_db, 2),
        "mean_motion_index": None
        if report.mean_motion_index is None
        else round(report.mean_motion_index, 4),
        "packets_parsed": report.packets_parsed,
        "packets_lost": report.packets_lost,
        "issues": report.issues,
    }
