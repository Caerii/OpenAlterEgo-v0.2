"""Bind hardware DSL specs to sim, collect, serve, and dataset generation."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..acquisition.packet import AfeSpec
from ..acquisition.simulate import SimConfig
from ..acquisition.virtual import VirtualBleSpec
from ..sim.dataset import DatasetConfig
from ..sim.stream import ScenarioConfig, SimStreamConfig
from .load import load_spec
from .resolve import resolve_all, resolve_sim_config, resolve_virtual_ble_spec
from .schema import HardwareSpec
from .validate import ValidationIssue, has_errors, validate_spec


def add_hw_spec_argument(ap: argparse.ArgumentParser, *, default: str = "") -> None:
    ap.add_argument(
        "--hw-spec",
        type=str,
        default=default,
        metavar="PRESET|PATH",
        help="Hardware DSL preset name or path to .oae.json (overrides fs/channels/preprocess/sim)",
    )


def load_hw_spec_optional(hw_spec: str, *, strict: bool = True) -> Optional[HardwareSpec]:
    raw = str(hw_spec or "").strip()
    if not raw:
        return None
    spec = load_spec(raw)
    issues = validate_spec(spec)
    if strict and has_errors(issues):
        lines = [i.format() for i in issues if i.severity == "error"]
        raise ValueError("hardware spec validation failed:\n" + "\n".join(lines))
    return spec


def load_hw_spec(hw_spec: str, *, strict: bool = True) -> HardwareSpec:
    spec = load_hw_spec_optional(hw_spec, strict=strict)
    if spec is None:
        raise ValueError("hw-spec is required")
    return spec


def merge_sim_config(
    spec: HardwareSpec,
    *,
    labels: Optional[List[str]] = None,
    p_event: Optional[float] = None,
    seed: Optional[int] = None,
    realtime_clock: Optional[bool] = None,
    sim_engine: Optional[str] = None,
    realism_preset: Optional[str] = None,
) -> SimConfig:
    """Build :class:`SimConfig` from hardware spec with optional CLI overrides."""
    cfg = resolve_sim_config(spec)
    if labels is not None:
        cfg.labels = list(labels)
    if p_event is not None:
        cfg.p_event = float(p_event)
    if seed is not None:
        cfg.seed = int(seed)
    if realtime_clock is not None:
        cfg.realtime_clock = bool(realtime_clock)
    if sim_engine is not None and str(sim_engine).strip():
        cfg.sim_engine = str(sim_engine)
    if realism_preset is not None and str(realism_preset).strip():
        cfg.realism_preset = str(realism_preset)
    return cfg


def sim_stream_config_from_spec(
    spec: HardwareSpec,
    scenario: ScenarioConfig,
    *,
    seed: int,
    realism_preset: Optional[str] = None,
) -> SimStreamConfig:
    cfg = merge_sim_config(spec, seed=seed, realism_preset=realism_preset)
    rp = str(cfg.realism_preset or spec.sim.realism)
    return SimStreamConfig(
        fs_hz=int(cfg.fs_hz),
        channels=int(cfg.channels),
        chunk_ms=int(cfg.chunk_ms),
        seed=int(cfg.seed),
        scenario=scenario,
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
        realtime_clock=False,
        realism_preset=rp,  # type: ignore[arg-type]
    )


def dataset_config_from_hw(
    spec: HardwareSpec,
    *,
    out_dir: str,
    duration_s: float,
    scenario: ScenarioConfig,
    seed: int,
    sim_engine: Optional[str] = None,
    realism_preset: Optional[str] = None,
    biophysical_noise_scale: Optional[float] = None,
    snr_target_db: Optional[float] = None,
    chunk_ms: Optional[int] = None,
) -> DatasetConfig:
    engine = str(sim_engine or spec.sim.engine)
    rp = realism_preset if realism_preset is not None else str(spec.sim.realism)
    ssc = sim_stream_config_from_spec(spec, scenario, seed=seed, realism_preset=rp)
    if chunk_ms is not None:
        ssc = replace(ssc, chunk_ms=int(chunk_ms))
    ns = float(biophysical_noise_scale if biophysical_noise_scale is not None else spec.sim.noise_scale)
    target = snr_target_db
    if target is None and spec.sim.snr_target_static_db is not None:
        target = float(spec.sim.snr_target_static_db)
    return DatasetConfig(
        out_dir=Path(out_dir),
        duration_s=float(duration_s),
        config=ssc,
        sim_engine=engine,  # type: ignore[arg-type]
        biophysical_noise_scale=ns,
        snr_target_db=target,
        montage_name=str(spec.electrodes.montage),
        realism_preset=rp if rp else None,
    )


def virtual_ble_from_hw(
    spec: HardwareSpec,
    *,
    seed: Optional[int] = None,
    loss_prob: Optional[float] = None,
    jitter_ms: Optional[float] = None,
    extra_latency_ms: Optional[float] = None,
) -> VirtualBleSpec:
    vs = resolve_virtual_ble_spec(spec)
    if seed is not None:
        vs = replace(vs, sim=replace(vs.sim, seed=int(seed)))
    link = vs.link
    if loss_prob is not None:
        link = replace(link, loss_prob=float(loss_prob))
    if jitter_ms is not None:
        link = replace(link, jitter_ms=float(jitter_ms))
    if extra_latency_ms is not None:
        link = replace(link, extra_latency_ms=float(extra_latency_ms))
    return replace(vs, link=link)


def preprocess_mode_from_spec(spec: HardwareSpec) -> str:
    return str(spec.preprocess.mode)


def hw_metadata_dict(spec: HardwareSpec) -> Dict[str, Any]:
    resolved = resolve_all(spec)
    s = spec
    return {
        "hardware_spec": {
            "name": s.name,
            "tier": s.tier,
            "description": s.description,
            "literature_refs": list(s.literature_refs),
            "electrode_type": s.electrodes.type,
            "montage": s.electrodes.montage,
            "montage_sites": list(resolved.montage_sites),
            "reference": s.electrodes.reference,
            "afe": {
                "part": s.afe.part,
                "fs_hz": s.afe.fs_hz,
                "channels": s.afe.channels,
                "gain": s.afe.gain,
            },
            "preprocess_mode": s.preprocess.mode,
            "emg_paradigm": resolved.emg_paradigm,
            "ble_device_name": s.ble.device_name,
        }
    }


def ble_afe_from_hw(spec: HardwareSpec) -> AfeSpec:
    return AfeSpec(
        adc_bits=int(spec.afe.adc_bits),
        vref_v=float(spec.afe.vref_v),
        gain=float(spec.afe.gain),
    )


def format_validation_issues(issues: List[ValidationIssue]) -> str:
    return "\n".join(i.format() for i in issues)
