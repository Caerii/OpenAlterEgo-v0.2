"""Acquisition sources (BLE, simulation, virtual links)."""

from .ble_client import BleSpec, stream_ble, stream_ble_chunks
from .simulate import SimConfig, stream_simulated, stream_simulated_chunks
from .virtual import VirtualBleSpec, stream_virtual_ble_chunks, virtual_notifications
from .packet import AfeSpec, PacketSpec, pack_oa_v1, parse_oa_v1, quantize_uV_to_counts, counts_to_uV
