"""Resolve hardware DSL → runtime configs (sim, virtual BLE, AFE)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from ..acquisition.packet import AfeSpec, PacketSpec
from ..acquisition.simulate import SimConfig
from ..acquisition.virtual import VirtualBleSpec
from ..sim.stream import SimStreamConfig
from ..sim.transport import LinkConfig
from .montages import get_montage
from .schema import HardwareSpec
from .validate import _resolve_emg_paradigm, validate_spec


@dataclass(frozen=True)
class ResolvedHardware:
    """Fully resolved view of a hardware spec for simulation and host scaling."""

    spec: HardwareSpec
    emg_paradigm: str
    montage_sites: tuple[str, ...]
    sim_config: SimConfig
    virtual_ble: VirtualBleSpec
    afe: AfeSpec
    preprocess_mode: str
    metadata: Dict[str, Any]


def resolve_sim_config(spec: HardwareSpec) -> SimConfig:
    paradigm = _resolve_emg_paradigm(spec)
    montage = get_montage(spec.electrodes.montage)
    ch = int(spec.afe.channels)
    return SimConfig(
        fs_hz=int(spec.afe.fs_hz),
        channels=ch,
        chunk_ms=int(spec.sim.chunk_ms),
        labels=list(spec.sim.labels),
        seed=int(spec.sim.seed),
        emg_paradigm=paradigm,
        noise_uV=float(spec.sim.noise_uV),
        drift_uV_per_s=float(spec.sim.drift_uV_per_s),
        crosstalk=float(spec.sim.crosstalk),
        ar1_phi=float(spec.sim.ar1_phi),
        ar1_innovation_scale=float(spec.sim.ar1_innovation_scale),
        line_noise_uV=float(spec.sim.line_noise_uV),
        mains_freq_hz=float(spec.sim.mains_freq_hz),
        noise_scale=float(spec.sim.noise_scale),
        montage_name=str(spec.electrodes.montage),
        realtime_clock=False,
        sim_engine=str(spec.sim.engine),
        realism_preset=str(spec.sim.realism),
    )


def resolve_virtual_ble_spec(spec: HardwareSpec) -> VirtualBleSpec:
    sim_cfg = resolve_sim_config(spec)
    sim_stream = SimStreamConfig(
        fs_hz=sim_cfg.fs_hz,
        channels=sim_cfg.channels,
        chunk_ms=sim_cfg.chunk_ms,
        emg_paradigm=sim_cfg.emg_paradigm,
        noise_uV=sim_cfg.noise_uV,
        drift_uV_per_s=sim_cfg.drift_uV_per_s,
        line_noise_uV=sim_cfg.line_noise_uV,
        mains_freq_hz=sim_cfg.mains_freq_hz,
        seed=sim_cfg.seed,
        realtime_clock=False,
        realism_preset=str(spec.sim.realism),
    )
    afe = AfeSpec(
        adc_bits=int(spec.afe.adc_bits),
        vref_v=float(spec.afe.vref_v),
        gain=float(spec.afe.gain),
    )
    return VirtualBleSpec(
        fs_hz=int(spec.afe.fs_hz),
        channels=int(spec.afe.channels),
        packet_format=spec.ble.packet_format,
        afe=afe,
        packet=PacketSpec(
            channels=int(spec.afe.channels),
            frames_per_packet=int(spec.ble.frames_per_packet),
        ),
        link=LinkConfig(
            loss_prob=float(spec.link.loss_prob),
            jitter_ms=float(spec.link.jitter_ms),
            extra_latency_ms=float(spec.link.extra_latency_ms),
            seed=int(spec.link.seed),
        ),
        sim=sim_stream,
    )


def resolve_afe_spec(spec: HardwareSpec) -> AfeSpec:
    return AfeSpec(
        adc_bits=int(spec.afe.adc_bits),
        vref_v=float(spec.afe.vref_v),
        gain=float(spec.afe.gain),
    )


def resolve_all(spec: HardwareSpec) -> ResolvedHardware:
    montage = get_montage(spec.electrodes.montage)
    paradigm = _resolve_emg_paradigm(spec)
    meta: Dict[str, Any] = {
        "hardware_name": spec.name,
        "tier": spec.tier,
        "literature_refs": list(spec.literature_refs),
        "electrode_type": spec.electrodes.type,
        "montage": spec.electrodes.montage,
        "reference": spec.electrodes.reference,
        "montage_sites": list(montage.sites),
        "montage_literature": montage.literature_ref,
        "emg_paradigm": paradigm,
        "preprocess_mode": spec.preprocess.mode,
        "ble_device_name": spec.ble.device_name,
        "validation_issues": [i.format() for i in validate_spec(spec)],
    }
    return ResolvedHardware(
        spec=spec,
        emg_paradigm=paradigm,
        montage_sites=tuple(montage.sites),
        sim_config=resolve_sim_config(spec),
        virtual_ble=resolve_virtual_ble_spec(spec),
        afe=resolve_afe_spec(spec),
        preprocess_mode=spec.preprocess.mode,
        metadata=meta,
    )


def resolved_to_jsonable(resolved: ResolvedHardware) -> Dict[str, Any]:
    """JSON-friendly summary for CLI ``hw resolve``."""
    s = resolved.spec
    return {
        "name": s.name,
        "tier": s.tier,
        "emg_paradigm": resolved.emg_paradigm,
        "preprocess_mode": resolved.preprocess_mode,
        "montage_sites": list(resolved.montage_sites),
        "afe": {
            "fs_hz": s.afe.fs_hz,
            "channels": s.afe.channels,
            "gain": s.afe.gain,
            "uV_per_count": resolved.afe.uV_per_count(),
        },
        "sim": {
            "engine": s.sim.engine,
            "realism": s.sim.realism,
            "labels": s.sim.labels,
        },
        "ble": {
            "packet_format": s.ble.packet_format,
            "frames_per_packet": s.ble.frames_per_packet,
            "link_loss_prob": s.link.loss_prob,
        },
        "metadata": resolved.metadata,
    }
