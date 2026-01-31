"""Virtual BLE source.

This simulates firmware -> BLE notifications -> parser, so you can stress-test:
- packet framing,
- loss/jitter,
- realtime pipeline.

It's intentionally NOT a real BLE peripheral (that would require OS-specific virtual BLE tools).
Instead, it emulates the *data path* at the byte level.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Literal, Optional

import numpy as np

from ..core.types import FrameChunk
from ..sim.stream import SimStream, SimStreamConfig
from ..sim.transport import LinkConfig, apply_link
from .packet import AfeSpec, PacketSpec, pack_oa_v1, parse_oa_v1, quantize_uV_to_counts


PacketFormat = Literal["raw_i16", "oa_v1"]


@dataclass
class VirtualBleSpec:
    fs_hz: int = 250
    channels: int = 8
    packet_format: PacketFormat = "oa_v1"

    afe: AfeSpec = field(default_factory=AfeSpec)
    packet: PacketSpec = field(default_factory=lambda: PacketSpec(channels=8, frames_per_packet=12))
    link: LinkConfig = field(default_factory=LinkConfig)

    sim: SimStreamConfig = field(default_factory=lambda: SimStreamConfig(fs_hz=250, channels=8, realtime_clock=False))

    # raw_i16 only:
    scale_uV_per_count: float = 1.0

    def __post_init__(self) -> None:
        # Keep nested configs consistent.
        self.sim.fs_hz = int(self.fs_hz)
        self.sim.channels = int(self.channels)
        self.packet = PacketSpec(channels=int(self.channels), frames_per_packet=int(self.packet.frames_per_packet))


async def virtual_notifications(spec: VirtualBleSpec) -> AsyncIterator[bytes]:
    """Yield notification payload bytes."""
    sim = SimStream(spec.sim)
    sample_index = 0

    while True:
        chunk = sim.next_chunk()  # no sleeping
        x_uV = chunk.samples
        counts = quantize_uV_to_counts(x_uV, afe=spec.afe)

        fpp = int(spec.packet.frames_per_packet)
        for i in range(0, counts.shape[0], fpp):
            block = counts[i : i + fpp, :]
            if block.shape[0] == 0:
                continue
            if spec.packet_format == "oa_v1":
                payload = pack_oa_v1(
                    block,
                    seq0=int(sample_index),
                    sample_index0=int(sample_index),
                    spec=spec.packet,
                )
            else:
                payload = block.astype("<i2", copy=False).tobytes(order="C")
            sample_index += int(block.shape[0])
            yield payload
            await asyncio.sleep(block.shape[0] / float(spec.fs_hz))


async def stream_virtual_ble_chunks(spec: VirtualBleSpec) -> AsyncIterator[FrameChunk]:
    """Yield FrameChunk objects after simulating link + parsing."""
    packets = apply_link(virtual_notifications(spec), cfg=spec.link)

    expected_sample_index: Optional[int] = None
    lost_samples_total = 0

    async for p in packets:
        now = time.time()
        if spec.packet_format == "oa_v1":
            frames_uV, info = parse_oa_v1(p, afe=spec.afe)
            sample_index0 = int(info.get("sample_index0", 0))
            if expected_sample_index is None:
                expected_sample_index = sample_index0 + frames_uV.shape[0]
            else:
                if sample_index0 > expected_sample_index:
                    lost_samples_total += int(sample_index0 - expected_sample_index)
                expected_sample_index = sample_index0 + frames_uV.shape[0]
            t0 = now - (frames_uV.shape[0] / float(spec.fs_hz))
            yield FrameChunk(
                samples=frames_uV,
                fs_hz=int(spec.fs_hz),
                t0=float(t0),
                seq0=sample_index0,
                meta={"lost_samples_total": int(lost_samples_total), **info},
            )
        else:
            data = np.frombuffer(p, dtype="<i2")
            if data.size % spec.channels != 0:
                continue
            frames = data.reshape(-1, spec.channels).astype(np.float32) * float(spec.scale_uV_per_count)
            t0 = now - (frames.shape[0] / float(spec.fs_hz))
            yield FrameChunk(samples=frames, fs_hz=int(spec.fs_hz), t0=float(t0), seq0=0, meta={})
