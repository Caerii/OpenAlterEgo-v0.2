"""Record data collection sessions from the simulator or BLE."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ..hardware.schema import HardwareSpec

from ..acquisition.simulate import open_sim_from_config
from ..sim.literature import DEFAULT_AR1_INNOVATION_SCALE
from ..sim.stream import ScenarioConfig, SimStream, SimStreamConfig
from .data_collection import DataCollectionSession
from .profile import PreprocessingMode


def _collect_from_sim_instance(
    *,
    sim: Any,
    output_dir: Path,
    user_id: str,
    duration_s: float,
    fs_hz: int,
    channels: int,
    preprocessing_mode: PreprocessingMode,
    session_extra: Optional[Dict[str, Any]] = None,
) -> Path:
    session = DataCollectionSession(
        user_id,
        int(fs_hz),
        int(channels),
        preprocessing_mode=preprocessing_mode,
    )
    target_samples = max(1, int(float(duration_s) * int(fs_hz)))
    collected = 0
    while collected < target_samples:
        ch = sim.next_chunk()
        session.add_chunk(ch.samples)
        collected += int(ch.samples.shape[0])

    total = session.get_total_samples()
    for ev in sim.events:
        s = max(0, int(ev.start_sample))
        e = min(total, int(ev.end_sample))
        if e > s:
            session.add_event(s, e, str(ev.label))

    return session.save(Path(output_dir), session_extra=session_extra)


def collect_from_sim(
    *,
    output_dir: Path,
    user_id: str,
    duration_s: float,
    fs_hz: int,
    channels: int,
    seed: int,
    labels: List[str],
    p_event: float,
    preprocessing_mode: PreprocessingMode,
    realtime_clock: bool = False,
    emg_paradigm: str = "semg_literature_clamped",
    ar1_innovation_scale: float | None = None,
    line_noise_uV: float = 0.0,
    mains_freq_hz: float = 60.0,
    session_extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Drain :class:`~openalterego.sim.stream.SimStream` into a session folder."""
    sc = ScenarioConfig(labels=list(labels), p_event=float(p_event))
    ar1 = DEFAULT_AR1_INNOVATION_SCALE if ar1_innovation_scale is None else float(ar1_innovation_scale)
    sim_cfg = SimStreamConfig(
        fs_hz=int(fs_hz),
        channels=int(channels),
        seed=int(seed),
        scenario=sc,
        emg_paradigm=str(emg_paradigm),
        ar1_innovation_scale=ar1,
        line_noise_uV=float(line_noise_uV),
        mains_freq_hz=float(mains_freq_hz),
        realtime_clock=bool(realtime_clock),
    )
    sim = SimStream(sim_cfg)
    return _collect_from_sim_instance(
        sim=sim,
        output_dir=output_dir,
        user_id=user_id,
        duration_s=duration_s,
        fs_hz=fs_hz,
        channels=channels,
        preprocessing_mode=preprocessing_mode,
        session_extra=session_extra,
    )


def collect_from_hw_spec(
    *,
    spec: HardwareSpec,
    output_dir: Path,
    user_id: str,
    duration_s: float,
    labels: Optional[List[str]] = None,
    p_event: float = 0.65,
    seed: int = 1337,
    realtime_clock: bool = False,
    sim_engine: Optional[str] = None,
    realism_preset: Optional[str] = None,
) -> Path:
    """Collect a session using parameters bound from a hardware DSL spec."""
    from ..hardware.bind import hw_metadata_dict, merge_sim_config

    cfg = merge_sim_config(
        spec,
        labels=labels or list(spec.sim.labels),
        p_event=p_event,
        seed=seed,
        realtime_clock=realtime_clock,
        sim_engine=sim_engine,
        realism_preset=realism_preset,
    )
    sim = open_sim_from_config(cfg)
    extra = hw_metadata_dict(spec)
    return _collect_from_sim_instance(
        sim=sim,
        output_dir=output_dir,
        user_id=user_id,
        duration_s=duration_s,
        fs_hz=int(cfg.fs_hz),
        channels=int(cfg.channels),
        preprocessing_mode=spec.preprocess.mode,  # type: ignore[arg-type]
        session_extra=extra,
    )


async def collect_from_ble_async(
    *,
    output_dir: Path,
    user_id: str,
    max_seconds: float,
    device_name: str,
    data_char_uuid: str,
    fs_hz: int = 250,
    channels: int = 8,
    packet_format: str = "raw_i16",
    preprocessing_mode: PreprocessingMode = "standard",
    scale_uV_per_count: float = 1.0,
    afe: Optional[Any] = None,
    session_extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Record BLE stream for a fixed duration (events.csv left empty for offline labeling)."""
    from ..acquisition.ble_client import BleSpec, stream_ble_chunks
    from ..acquisition.packet import AfeSpec

    spec = BleSpec(
        device_name=str(device_name),
        data_char_uuid=str(data_char_uuid),
        fs_hz=int(fs_hz),
        channels=int(channels),
        packet_format=packet_format,  # type: ignore[arg-type]
        scale_uV_per_count=float(scale_uV_per_count),
        afe=afe if afe is not None else AfeSpec(),
    )
    session = DataCollectionSession(
        user_id,
        int(fs_hz),
        int(channels),
        preprocessing_mode=preprocessing_mode,
    )
    deadline = asyncio.get_event_loop().time() + float(max_seconds)
    async for chunk in stream_ble_chunks(spec):
        session.add_chunk(chunk.samples)
        if asyncio.get_event_loop().time() >= deadline:
            break
    return session.save(Path(output_dir), session_extra=session_extra)


def collect_from_ble(
    *,
    output_dir: Path,
    user_id: str,
    max_seconds: float,
    device_name: str,
    data_char_uuid: str,
    fs_hz: int = 250,
    channels: int = 8,
    packet_format: str = "raw_i16",
    preprocessing_mode: PreprocessingMode = "standard",
    scale_uV_per_count: float = 1.0,
) -> Path:
    return asyncio.run(
        collect_from_ble_async(
            output_dir=output_dir,
            user_id=user_id,
            max_seconds=max_seconds,
            device_name=device_name,
            data_char_uuid=data_char_uuid,
            fs_hz=fs_hz,
            channels=channels,
            packet_format=packet_format,
            preprocessing_mode=preprocessing_mode,
            scale_uV_per_count=scale_uV_per_count,
        )
    )
