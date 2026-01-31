"""BLE acquisition (usable, but still a skeleton).

What you get here:
- scanning + connect by device name
- notification subscription
- parsing either a *raw int16 interleaved* format or the OpenAlterEgo v1 framed packet
- async generator that yields either numpy arrays or FrameChunk objects

What you still need to do for *your* board:
- UUIDs
- ensuring your firmware's packet format matches one of the parsers (or add your own)

Optional dependency
-------------------
BLE support uses the `bleak` package. Simulation-only workflows should not require it.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import AsyncIterator, Literal, Optional

import numpy as np

from ..core.types import FrameChunk
from .packet import AfeSpec, parse_oa_v1


PacketFormat = Literal["raw_i16", "oa_v1"]


@dataclass
class BleSpec:
    device_name: str
    data_char_uuid: str

    fs_hz: int = 250
    channels: int = 8

    # raw_i16: scale to microvolts by a known factor
    scale_uV_per_count: float = 1.0

    packet_format: PacketFormat = "raw_i16"

    # used for oa_v1 scaling
    afe: AfeSpec = AfeSpec()


def _parse_int16_interleaved(payload: bytes, channels: int, scale: float) -> np.ndarray:
    data = np.frombuffer(payload, dtype="<i2")  # little-endian int16
    if data.size % channels != 0:
        raise ValueError(f"payload size {data.size} not divisible by channels={channels}")
    frames = data.reshape(-1, channels).astype(np.float32) * float(scale)
    return frames


async def stream_ble_chunks(spec: BleSpec) -> AsyncIterator[FrameChunk]:
    """Async generator that yields :class:`~openalterego.core.types.FrameChunk`."""
    try:
        from bleak import BleakClient, BleakScanner  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "BLE support requires the 'bleak' package. Install it with: pip install bleak"
        ) from e

    dev = await BleakScanner.find_device_by_filter(lambda d, ad: d.name == spec.device_name)
    if dev is None:
        raise RuntimeError(f"BLE device not found: {spec.device_name!r}")

    queue: asyncio.Queue[FrameChunk] = asyncio.Queue()
    expected_sample_index: Optional[int] = None
    lost_samples_total = 0

    def _on_notify(_: int, payload: bytearray):
        nonlocal expected_sample_index, lost_samples_total
        try:
            now = time.time()
            if spec.packet_format == "oa_v1":
                frames_uV, info = parse_oa_v1(bytes(payload), afe=spec.afe)
                sample_index0 = int(info.get("sample_index0", 0))
                if expected_sample_index is None:
                    expected_sample_index = sample_index0 + frames_uV.shape[0]
                else:
                    if sample_index0 > expected_sample_index:
                        lost_samples_total += int(sample_index0 - expected_sample_index)
                    expected_sample_index = sample_index0 + frames_uV.shape[0]
                t0 = now - (frames_uV.shape[0] / float(spec.fs_hz))
                chunk = FrameChunk(
                    samples=frames_uV,
                    fs_hz=int(spec.fs_hz),
                    t0=float(t0),
                    seq0=sample_index0,
                    meta={"lost_samples_total": int(lost_samples_total), **info},
                )
            else:
                frames_uV = _parse_int16_interleaved(bytes(payload), spec.channels, spec.scale_uV_per_count)
                t0 = now - (frames_uV.shape[0] / float(spec.fs_hz))
                chunk = FrameChunk(samples=frames_uV, fs_hz=int(spec.fs_hz), t0=float(t0), seq0=0, meta={})

            queue.put_nowait(chunk)
        except Exception as e:
            # Don't crash the BLE callback
            print(f"[ble notify parse error] {e}")

    async with BleakClient(dev) as client:
        await client.start_notify(spec.data_char_uuid, _on_notify)
        try:
            while True:
                chunk = await queue.get()
                yield chunk
        finally:
            await client.stop_notify(spec.data_char_uuid)


async def stream_ble(spec: BleSpec) -> AsyncIterator[np.ndarray]:
    """Legacy helper that yields only sample arrays."""
    async for chunk in stream_ble_chunks(spec):
        yield chunk.samples
