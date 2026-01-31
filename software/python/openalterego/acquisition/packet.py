"""Binary packet formats for OpenAlterEgo acquisition.

Why?
----
BLE notifications are just byte blobs. You *need* a framing format that survives:
- packet reordering/loss,
- variable BLE MTU sizes,
- future firmware changes.

The v1 format here is intentionally simple and designed for microcontrollers.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


MAGIC = b"OA"
VERSION = 1
HEADER_BYTES = 20  # keep in sync with struct format below


@dataclass(frozen=True)
class AfeSpec:
    """Analog front-end + ADC model (used for scaling)."""

    adc_bits: int = 16
    vref_v: float = 2.4
    gain: float = 24.0

    def uV_per_count(self) -> float:
        fullscale_in_v = self.vref_v / self.gain
        max_count = (2 ** (self.adc_bits - 1)) - 1
        return (fullscale_in_v / max_count) * 1e6


@dataclass(frozen=True)
class PacketSpec:
    channels: int
    frames_per_packet: int = 12


def quantize_uV_to_counts(x_uV: np.ndarray, *, afe: AfeSpec) -> np.ndarray:
    """Convert microvolts to ADC counts (clipped)."""
    if x_uV.ndim != 2:
        raise ValueError(f"x_uV must be 2D (time, channels), got {x_uV.shape}")
    scale = afe.uV_per_count()
    counts = np.round(x_uV / scale).astype(np.int64)
    lo = -(2 ** (afe.adc_bits - 1))
    hi = (2 ** (afe.adc_bits - 1)) - 1
    counts = np.clip(counts, lo, hi).astype(np.int16)
    return counts


def counts_to_uV(counts: np.ndarray, *, afe: AfeSpec) -> np.ndarray:
    if counts.ndim != 2:
        raise ValueError(f"counts must be 2D, got {counts.shape}")
    return counts.astype(np.float32) * float(afe.uV_per_count())


def pack_oa_v1(frames_i16: np.ndarray, *, seq0: int, sample_index0: int, spec: PacketSpec) -> bytes:
    """Pack frames into an OpenAlterEgo v1 binary packet.

    Header (little endian):
    - magic[2] = b"OA"
    - version u8
    - channels u8
    - frames u16
    - flags u16 (reserved)
    - seq0 u32
    - sample_index0 u64

    Payload:
    - int16 samples, interleaved by channel
    """
    if frames_i16.ndim != 2 or frames_i16.shape[1] != spec.channels:
        raise ValueError(f"frames_i16 must have shape (time, {spec.channels}), got {frames_i16.shape}")
    frames = int(frames_i16.shape[0])
    if frames <= 0:
        raise ValueError("frames_i16 must contain at least one frame")
    if frames > 65535:
        raise ValueError("too many frames for u16")

    flags = 0
    header = struct.pack(
        "<2sBBHHIQ",
        MAGIC,
        VERSION,
        int(spec.channels) & 0xFF,
        frames & 0xFFFF,
        flags & 0xFFFF,
        int(seq0) & 0xFFFFFFFF,
        int(sample_index0) & 0xFFFFFFFFFFFFFFFF,
    )
    payload = frames_i16.astype("<i2", copy=False).tobytes(order="C")
    return header + payload


def parse_oa_v1(payload: bytes, *, afe: Optional[AfeSpec] = None) -> Tuple[np.ndarray, dict]:
    """Parse OpenAlterEgo v1 packet.

    Returns
    -------
    frames:
        (time, channels) float32 (microvolts) if afe is provided, else int16 counts.
    info:
        dict with fields: version, channels, frames, seq0, sample_index0, flags
    """
    if len(payload) < HEADER_BYTES:
        raise ValueError("payload too short")

    magic, ver, ch, frames, flags, seq0, sample_index0 = struct.unpack("<2sBBHHIQ", payload[:HEADER_BYTES])
    if magic != MAGIC:
        raise ValueError(f"bad magic {magic!r}")
    if ver != VERSION:
        raise ValueError(f"unsupported version {ver}")
    if frames <= 0:
        raise ValueError("frames must be >0")

    expected = HEADER_BYTES + frames * ch * 2
    if len(payload) != expected:
        raise ValueError(f"length mismatch: got {len(payload)} bytes, expected {expected}")

    data = np.frombuffer(payload[HEADER_BYTES:], dtype="<i2")
    if data.size != frames * ch:
        raise ValueError("data size mismatch")
    arr = data.reshape(frames, ch)

    if afe is not None:
        out = counts_to_uV(arr, afe=afe)
    else:
        out = arr

    info = {
        "version": int(ver),
        "channels": int(ch),
        "frames": int(frames),
        "flags": int(flags),
        "seq0": int(seq0),
        "sample_index0": int(sample_index0),
    }
    return out, info
