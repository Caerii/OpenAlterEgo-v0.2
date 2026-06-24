"""Synthetic dataset generation utilities."""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd

from .biophysical.dataset_meta import biophysical_block_for_meta
from .biophysical.stream import BiophysicalSimStream, BiophysicalSimStreamConfig
from .literature import LITERATURE_MODEL_VERSION, resolve_sim_token_band
from .stream import SimStream, SimStreamConfig


@dataclass
class DatasetConfig:
    out_dir: Path
    duration_s: float = 60.0
    config: SimStreamConfig = field(default_factory=SimStreamConfig)
    sim_engine: Literal["heuristic", "biophysical"] = "biophysical"
    biophysical: Optional[BiophysicalSimStreamConfig] = None
    # Applied when ``biophysical`` is None and ``sim_engine == "biophysical"`` (SNR regime sweeps).
    biophysical_noise_scale: float = 1.0
    # Auto-tune ``biophysical_noise_scale`` toward this session SNR (dB) before generation.
    snr_target_db: Optional[float] = None
    snr_motion_target_db: Optional[float] = None
    montage_name: Optional[str] = None
    # If set, drop events shorter than this (helps avoid ultra-short segments)
    min_event_s: float = 0.15
    min_phoneme_s: float = 0.02
    write_phonemes_csv: bool = True
    # If set, overrides :attr:`BiophysicalSimStreamConfig.realism_preset` / :attr:`SimStreamConfig.realism_preset`.
    realism_preset: Optional[str] = None
    phone_templates_path: Optional[str] = None

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)


def generate_dataset(ds: DatasetConfig) -> Path:
    """Generate a session-like folder with signals.npy + events.csv."""
    ds.out_dir.mkdir(parents=True, exist_ok=True)

    cfg_eff = ds.config
    if ds.realism_preset is not None:
        cfg_eff = replace(ds.config, realism_preset=str(ds.realism_preset))  # type: ignore[arg-type]

    noise_scale = float(ds.biophysical_noise_scale)
    lf_snr_scale = 1.0
    motion_burst_scale = 1.0
    snr_calibration_meta: Optional[dict] = None
    eff_realism = str(ds.realism_preset or cfg_eff.realism_preset)
    montage = str(ds.montage_name).strip() or None

    if ds.sim_engine == "biophysical" and (
        ds.snr_target_db is not None or ds.snr_motion_target_db is not None
    ):
        if ds.snr_motion_target_db is not None:
            from .snr_calibration import TANG_SNR_MOTION_DB, TANG_SNR_STATIC_DB, tune_snr_regimes

            cal = tune_snr_regimes(
                fs_hz=int(cfg_eff.fs_hz),
                channels=int(cfg_eff.channels),
                emg_paradigm=str(cfg_eff.emg_paradigm),
                realism_preset=eff_realism,
                static_target_db=float(ds.snr_target_db or TANG_SNR_STATIC_DB),
                motion_target_db=float(ds.snr_motion_target_db),
                electrode_noise_uV=float(cfg_eff.noise_uV),
                line_noise_uV=float(cfg_eff.line_noise_uV),
                drift_uV_per_s=float(cfg_eff.drift_uV_per_s),
                crosstalk=float(cfg_eff.crosstalk),
                montage_name=montage,
                seed=int(cfg_eff.seed),
            )
            noise_scale = float(cal.lf_snr_scale)
            lf_snr_scale = float(cal.lf_snr_scale)
            motion_burst_scale = float(cal.motion_burst_scale)
            snr_calibration_meta = cal.to_dict()
        else:
            from .snr_calibration import tune_noise_scale

            cal = tune_noise_scale(
                fs_hz=int(cfg_eff.fs_hz),
                channels=int(cfg_eff.channels),
                emg_paradigm=str(cfg_eff.emg_paradigm),
                realism_preset=eff_realism,
                target_snr_db=float(ds.snr_target_db),
                electrode_noise_uV=float(cfg_eff.noise_uV),
                line_noise_uV=float(cfg_eff.line_noise_uV),
                drift_uV_per_s=float(cfg_eff.drift_uV_per_s),
                crosstalk=float(cfg_eff.crosstalk),
                montage_name=montage,
                seed=int(cfg_eff.seed),
            )
            noise_scale = float(cal.lf_snr_scale)
            lf_snr_scale = float(cal.lf_snr_scale)
            snr_calibration_meta = {
                "target_snr_db": cal.target_snr_db,
                "measured_snr_db": cal.measured_snr_db,
                "measured_motion_index": cal.measured_motion_index,
                "lf_snr_scale": cal.lf_snr_scale,
                "iterations": cal.iterations,
            }

    if ds.sim_engine == "biophysical":
        if ds.biophysical is not None:
            bcfg = ds.biophysical
            if ds.realism_preset is not None:
                bcfg = replace(bcfg, realism_preset=str(ds.realism_preset))
            if ds.phone_templates_path is not None:
                bcfg = replace(bcfg, phone_templates_path=str(ds.phone_templates_path))
        else:
            bcfg = BiophysicalSimStreamConfig(
                fs_hz=int(cfg_eff.fs_hz),
                channels=int(cfg_eff.channels),
                chunk_ms=int(cfg_eff.chunk_ms),
                seed=int(cfg_eff.seed),
                scenario=cfg_eff.scenario,
                realtime_clock=False,
                emg_paradigm=str(cfg_eff.emg_paradigm),
                token_band_hz=cfg_eff.token_band_hz,
                electrode_noise_uV=float(cfg_eff.noise_uV),
                drift_uV_per_s=float(cfg_eff.drift_uV_per_s),
                crosstalk=float(cfg_eff.crosstalk),
                ar1_phi=float(cfg_eff.ar1_phi),
                ar1_innovation_scale=float(cfg_eff.ar1_innovation_scale),
                line_noise_uV=float(cfg_eff.line_noise_uV),
                mains_freq_hz=float(cfg_eff.mains_freq_hz),
                noise_scale=float(noise_scale),
                lf_snr_scale=float(lf_snr_scale),
                motion_burst_scale=float(motion_burst_scale),
                montage_name=montage,
                realism_preset=eff_realism,  # type: ignore[arg-type]
                phone_templates_path=str(ds.phone_templates_path) if ds.phone_templates_path else None,
            )
        sim = BiophysicalSimStream(bcfg)
        fs_hz = int(bcfg.fs_hz)
        ch = int(bcfg.channels)
        seed = int(bcfg.seed)
        labels = list(bcfg.scenario.labels)
        emg_paradigm = str(bcfg.emg_paradigm)
        token_band_key = bcfg.token_band_hz
        eff_realism = str(bcfg.realism_preset)
        sim_meta_extra = {
            "sim_engine": "biophysical",
            "biophysical": biophysical_block_for_meta(bcfg),
            "realism_preset": eff_realism,
            "montage_name": montage,
            "noise_scale": float(noise_scale),
            "lf_snr_scale": float(lf_snr_scale),
            "motion_burst_scale": float(motion_burst_scale),
        }
        if snr_calibration_meta is not None:
            sim_meta_extra["snr_calibration"] = snr_calibration_meta
    else:
        sim = SimStream(cfg_eff)
        fs_hz = int(cfg_eff.fs_hz)
        ch = int(cfg_eff.channels)
        seed = int(cfg_eff.seed)
        labels = list(cfg_eff.scenario.labels)
        emg_paradigm = str(cfg_eff.emg_paradigm)
        token_band_key = cfg_eff.token_band_hz
        eff_realism = str(cfg_eff.realism_preset)
        sim_meta_extra = {"sim_engine": "heuristic", "realism_preset": eff_realism}

    total_samples = int(ds.duration_s * fs_hz)

    signals = np.zeros((total_samples, ch), dtype=np.float32)
    i = 0
    while i < total_samples:
        chunk = sim.next_chunk()
        x = chunk.samples
        n = x.shape[0]
        take = min(n, total_samples - i)
        signals[i : i + take, :] = x[:take, :]
        i += take

    rows = []
    has_trial = False
    scripted = bool(getattr(cfg_eff.scenario, "scripted_schedule", None))
    for ev in sim.events:
        dur_s = (ev.end_sample - ev.start_sample) / float(fs_hz)
        if dur_s < ds.min_event_s:
            continue
        if scripted and ev.trial_id is None:
            continue
        row = {"start_sample": int(ev.start_sample), "end_sample": int(ev.end_sample), "label": str(ev.label)}
        if ev.trial_id is not None:
            row["trial_id"] = int(ev.trial_id)
            row["word_idx"] = int(ev.word_idx) if ev.word_idx is not None else 0
            has_trial = True
        rows.append(row)

    cols = ["start_sample", "end_sample", "label"]
    if has_trial:
        cols.extend(["trial_id", "word_idx"])
    events = pd.DataFrame(rows, columns=cols)
    signals_path = ds.out_dir / "signals.npy"
    events_path = ds.out_dir / "events.csv"

    np.save(signals_path, signals)
    events.to_csv(events_path, index=False)

    phone_rows: list[dict] = []
    if (
        ds.write_phonemes_csv
        and ds.sim_engine == "biophysical"
        and hasattr(sim, "phoneme_events")
        and getattr(sim, "drive_uses_phonemes", False)
    ):
        for seg in sim.phoneme_events:
            dur_s = (seg.end_sample - seg.start_sample) / float(fs_hz)
            if dur_s < ds.min_phoneme_s:
                continue
            phone_rows.append(
                {
                    "start_sample": int(seg.start_sample),
                    "end_sample": int(seg.end_sample),
                    "phone": str(seg.phone),
                    "word": str(seg.word),
                }
            )
        if phone_rows:
            pd.DataFrame(
                phone_rows,
                columns=["start_sample", "end_sample", "phone", "word"],
            ).to_csv(ds.out_dir / "phonemes.csv", index=False)

    # Compute signal quality metrics for metadata
    from ..dsp.quality import assess_signal_quality
    
    token_band = resolve_sim_token_band(fs_hz, emg_paradigm, token_band_key)
    quality_metrics = assess_signal_quality(
        signals,
        fs_hz=float(fs_hz),
        signal_band_hz=token_band,
        noise_band_hz=(0.5, 5.0),
        low_freq_cutoff_hz=5.0,
        axis=0,
    )
    
    phone_inventory: list[str] = []
    if ds.sim_engine == "biophysical" and hasattr(sim, "_phone_inventory"):
        phone_inventory = [str(p) for p in getattr(sim, "_phone_inventory", [])]

    meta = {
        "fs_hz": int(fs_hz),
        "channels": int(ch),
        "duration_s": float(ds.duration_s),
        "labels": labels,
        "phones": phone_inventory,
        "drive_mode": str(ds.config.scenario.drive_mode),
        "seed": int(seed),
        "emg_paradigm": emg_paradigm,
        "literature_model": LITERATURE_MODEL_VERSION,
        **sim_meta_extra,
        "sim_config": {
            "noise_uV": float(ds.config.noise_uV),
            "drift_uV_per_s": float(ds.config.drift_uV_per_s),
            "crosstalk": float(ds.config.crosstalk),
            "ar1_phi": float(ds.config.ar1_phi),
            "ar1_innovation_scale": float(ds.config.ar1_innovation_scale),
            "line_noise_uV": float(ds.config.line_noise_uV),
            "mains_freq_hz": float(ds.config.mains_freq_hz),
            "token_band_hz": list(token_band),
            "token_amplitude_uV": float(ds.config.token_amplitude_uV),
            "realism_preset": eff_realism,
        },
        "quality_metrics": {
            "snr_db": quality_metrics.snr_db,
            "motion_index": float(quality_metrics.motion_index),
            "baseline_wander": float(quality_metrics.baseline_wander),
            "signal_power": float(quality_metrics.signal_power),
            "noise_power": float(quality_metrics.noise_power),
        },
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (ds.out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return ds.out_dir


def _generate_dataset_shard(ds: DatasetConfig, shard_id: int) -> Path:
    shard_dir = Path(ds.out_dir) / f"shard_{shard_id:03d}"
    cfg = replace(
        ds,
        out_dir=shard_dir,
        config=replace(ds.config, seed=int(ds.config.seed) + int(shard_id)),
    )
    if cfg.biophysical is not None:
        cfg = replace(
            cfg,
            biophysical=replace(cfg.biophysical, seed=int(cfg.biophysical.seed) + int(shard_id)),
        )
    return generate_dataset(cfg)


def generate_dataset_shards(
    ds: DatasetConfig,
    *,
    n_shards: int = 2,
    workers: int = 2,
) -> list[Path]:
    """Generate ``n_shards`` session folders in parallel under ``ds.out_dir``."""
    n_shards = max(1, int(n_shards))
    workers = max(1, min(int(workers), n_shards))
    parent = Path(ds.out_dir)
    parent.mkdir(parents=True, exist_ok=True)

    if n_shards == 1:
        return [_generate_dataset_shard(ds, 0)]

    paths: list[Path] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_generate_dataset_shard, ds, i) for i in range(n_shards)]
        for fut in as_completed(futures):
            paths.append(fut.result())
    return sorted(paths)
