"""Simulation utilities (synthetic signals, hardware-ish packetization, transport)."""

from .stream import SimStream, SimStreamConfig, ScenarioConfig, SimEvent
from .dataset import DatasetConfig, generate_dataset
from .hardware import AfeSpec, PacketSpec, pack_oa_v1, parse_oa_v1, quantize_uV_to_counts, counts_to_uV
from .transport import LinkConfig, apply_link
