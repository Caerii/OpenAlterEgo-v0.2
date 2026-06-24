"""Compare Gowda-shaped sim probes to real OSF event statistics."""

from __future__ import annotations

import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ...dsp.emg_config import get_filter_spec_for_mode
from ...dsp.filters import preprocess_basic
from ...dsp.quality import detect_motion_artifacts
from ...ml.data_split import gowda_official_train_val_test_indices
from ...ml.datasets.events import load_gowda_events, sanitize_trial_events
from ..dataset import generate_dataset
from ..literature import resolve_sim_token_band
from ..scenarios.gowda_small_vocab import (
    GOWDA_CHANNELS,
    GOWDA_FS_HZ,
    build_gowda_dataset_config,
)


@dataclass(frozen=True)
class RealismVariant:
    """One realism preset + optional Tang SNR calibration targets."""

    tag: str
    preset: str
    snr_target_db: Optional[float] = None
    snr_motion_target_db: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SegmentStats:
    n_segments: int
    median_snr_db: Optional[float]
    snr_p10: Optional[float]
    snr_p90: Optional[float]
    median_motion_index: float
    channel_rms: np.ndarray
    corr_offdiag_mean: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_segments": int(self.n_segments),
            "median_snr_db": self.median_snr_db,
            "snr_p10": self.snr_p10,
            "snr_p90": self.snr_p90,
            "median_motion_index": float(self.median_motion_index),
            "channel_rms": [float(x) for x in self.channel_rms.tolist()],
            "corr_offdiag_mean": float(self.corr_offdiag_mean),
        }


def default_realism_variants() -> List[RealismVariant]:
    """Preset ladder + Tang SNR sweeps used in realism ablations."""
    return [
        RealismVariant("off_raw", preset="off"),
        RealismVariant("wearable_cal", preset="wearable", snr_target_db=18.9, snr_motion_target_db=12.7),
        RealismVariant("tang_cal", preset="tang", snr_target_db=18.9, snr_motion_target_db=12.7),
        RealismVariant("field_cal", preset="field", snr_target_db=18.9, snr_motion_target_db=12.7),
        RealismVariant("tang_nocal", preset="tang"),
        RealismVariant("tang_lo_snr", preset="tang", snr_target_db=15.0, snr_motion_target_db=10.0),
        RealismVariant("tang_hi_snr", preset="tang", snr_target_db=22.0, snr_motion_target_db=15.0),
    ]


def _gowda_signal_band(fs_hz: float) -> tuple[float, float]:
    spec = get_filter_spec_for_mode("gowda", float(fs_hz), notch_hz=60.0)
    lo, hi = spec.bandpass_hz
    return float(lo), float(hi)


def _segment_snr_db_time_domain(
    x: np.ndarray,
    fs_hz: float,
    signal_band_hz: tuple[float, float],
) -> Optional[float]:
    """Band-power SNR robust at 5 kHz (Welch 0.5–5 Hz noise band is too coarse)."""
    from scipy import signal as sig

    if x.shape[0] < 32:
        return None
    nyq = float(fs_hz) / 2.0
    lo_n, hi_n = 0.5, min(5.0, nyq * 0.95)
    hi_s = min(float(signal_band_hz[1]), nyq * 0.99)
    if hi_n <= lo_n or hi_s <= float(signal_band_hz[0]):
        return None
    try:
        sos_s = sig.butter(
            4,
            [float(signal_band_hz[0]), hi_s],
            btype="band",
            fs=float(fs_hz),
            output="sos",
        )
        sos_n = sig.butter(4, [lo_n, hi_n], btype="band", fs=float(fs_hz), output="sos")
    except ValueError:
        return None
    xd = x.astype(np.float64)
    sig_bp = sig.sosfilt(sos_s, xd, axis=0)
    noise_bp = sig.sosfilt(sos_n, xd, axis=0)
    sp = np.mean(sig_bp * sig_bp, axis=0)
    npw = np.maximum(np.mean(noise_bp * noise_bp, axis=0), 1e-12)
    snr_ch = 10.0 * np.log10(sp / npw)
    finite = snr_ch[np.isfinite(snr_ch)]
    return float(np.median(finite)) if finite.size else None


def _segment_stats_from_events(
    signals: np.ndarray,
    events: pd.DataFrame,
    *,
    fs_hz: float,
    max_segments: int = 400,
    seed: int = 0,
    preprocess_mode: str = "gowda",
    snr_signal_band_hz: Optional[tuple[float, float]] = None,
) -> SegmentStats:
    rows = events.reset_index(drop=True)
    if rows.empty:
        return SegmentStats(
            n_segments=0,
            median_snr_db=None,
            snr_p10=None,
            snr_p90=None,
            median_motion_index=0.0,
            channel_rms=np.zeros((signals.shape[1],), dtype=np.float64),
            corr_offdiag_mean=0.0,
        )

    rng = np.random.default_rng(int(seed))
    idx = np.arange(len(rows))
    if len(idx) > int(max_segments):
        rng.shuffle(idx)
        idx = idx[: int(max_segments)]
    rows = rows.iloc[idx]

    band = snr_signal_band_hz or _gowda_signal_band(fs_hz)
    snrs: list[float] = []
    motions: list[float] = []
    rms_acc = np.zeros((signals.shape[1],), dtype=np.float64)
    corr_sum = 0.0
    n_corr = 0

    for row in rows.itertuples(index=False):
        s0 = int(row.start_sample)
        s1 = int(row.end_sample)
        raw = np.asarray(signals[s0:s1, :], dtype=np.float32)
        if raw.shape[0] < 16:
            continue
        snr = _segment_snr_db_time_domain(raw, float(fs_hz), band)
        if snr is not None:
            snrs.append(float(snr))
        motion_index, _ = detect_motion_artifacts(raw, float(fs_hz), low_freq_cutoff_hz=5.0, axis=0)
        motions.append(float(motion_index))
        proc = preprocess_basic(raw, fs_hz=float(fs_hz), mode=preprocess_mode)  # type: ignore[arg-type]
        rms_acc += np.sqrt(np.mean(proc * proc, axis=0))
        if proc.shape[0] >= 32 and proc.shape[1] >= 2:
            c = np.corrcoef(proc.T)
            off = c[np.triu_indices(c.shape[0], k=1)]
            if off.size:
                corr_sum += float(np.mean(np.abs(off)))
                n_corr += 1

    n_seg = max(1, int(len(motions)))
    snr_arr = np.asarray(snrs, dtype=np.float64) if snrs else None
    return SegmentStats(
        n_segments=int(len(motions)),
        median_snr_db=float(np.median(snr_arr)) if snr_arr is not None and snr_arr.size else None,
        snr_p10=float(np.percentile(snr_arr, 10)) if snr_arr is not None and snr_arr.size else None,
        snr_p90=float(np.percentile(snr_arr, 90)) if snr_arr is not None and snr_arr.size else None,
        median_motion_index=float(np.median(motions)) if motions else 0.0,
        channel_rms=rms_acc / float(n_seg),
        corr_offdiag_mean=float(corr_sum / n_corr) if n_corr else 0.0,
    )


def real_gowda_baseline_stats(
    real_dir: Path,
    *,
    max_events: int = 400,
    seed: int = 0,
    split: str = "train",
) -> SegmentStats:
    """Event-segment statistics on real Gowda OSF session (gowda band)."""
    real_dir = Path(real_dir)
    signals = np.load(real_dir / "signals.npy", mmap_mode="r")
    events = load_gowda_events(real_dir)
    trial_ids = events["trial_id"].astype(int).values
    tr_idx, _, _ = gowda_official_train_val_test_indices(trial_ids)
    if str(split) == "train":
        events = events.iloc[tr_idx].reset_index(drop=True)
    elif str(split) == "all":
        events = events.reset_index(drop=True)
    else:
        raise ValueError(f"unsupported split {split!r}")
    meta = __import__("json").loads((real_dir / "meta.json").read_text(encoding="utf-8"))
    fs_hz = float(meta.get("fs_hz", GOWDA_FS_HZ))
    return _segment_stats_from_events(
        signals, events, fs_hz=fs_hz, max_segments=max_events, seed=seed, preprocess_mode="gowda"
    )


def sim_gowda_variant_stats(
    variant: RealismVariant,
    *,
    probe_trials: int = 8,
    seed: int = 1337,
    out_dir: Optional[Path] = None,
) -> tuple[SegmentStats, dict[str, Any]]:
    """Generate a short Gowda-shaped sim corpus and return event stats + meta."""
    probe_trials = max(2, int(probe_trials))
    owns_dir = out_dir is None
    if owns_dir:
        td = tempfile.mkdtemp(prefix="oae_realism_probe_")
        out_dir = Path(td)
    else:
        out_dir = Path(out_dir)

    ds = build_gowda_dataset_config(
        out_dir,
        n_trials=int(probe_trials),
        seed=int(seed),
        realism=str(variant.preset),
        snr_target_db=variant.snr_target_db,
        snr_motion_target_db=variant.snr_motion_target_db,
    )
    session = generate_dataset(ds)
    signals = np.load(session / "signals.npy", mmap_mode="r")
    events = sanitize_trial_events(pd.read_csv(session / "events.csv"))
    meta = __import__("json").loads((session / "meta.json").read_text(encoding="utf-8"))
    fs_hz = float(meta.get("fs_hz", GOWDA_FS_HZ))
    emg_paradigm = str(meta.get("biophysical", {}).get("emg_paradigm", "semg_literature_clamped"))
    snr_band = resolve_sim_token_band(int(fs_hz), emg_paradigm, None)
    stats = _segment_stats_from_events(
        signals,
        events,
        fs_hz=fs_hz,
        max_segments=10_000,
        seed=seed,
        preprocess_mode="gowda",
        snr_signal_band_hz=snr_band,
    )
    extra = {
        "session_dir": str(session),
        "meta": meta,
        "variant": variant.to_dict(),
        "probe_trials": int(probe_trials),
    }
    return stats, extra


def match_score(sim: SegmentStats, real: SegmentStats) -> dict[str, float]:
    """Lower total is a closer sim→real match."""
    parts: dict[str, float] = {}
    if sim.median_snr_db is not None and real.median_snr_db is not None:
        parts["snr_db"] = abs(float(sim.median_snr_db) - float(real.median_snr_db)) / 5.0
    else:
        parts["snr_db"] = 5.0

    sim_r = sim.channel_rms / (float(np.linalg.norm(sim.channel_rms)) + 1e-9)
    real_r = real.channel_rms / (float(np.linalg.norm(real.channel_rms)) + 1e-9)
    n = min(sim_r.shape[0], real_r.shape[0])
    parts["channel_rms"] = float(np.linalg.norm(sim_r[:n] - real_r[:n]))

    parts["corr"] = abs(float(sim.corr_offdiag_mean) - float(real.corr_offdiag_mean))
    parts["motion"] = abs(float(sim.median_motion_index) - float(real.median_motion_index))
    parts["total"] = float(sum(parts.values()))
    return parts


def parse_variant_tags(tags: Optional[Sequence[str]]) -> List[RealismVariant]:
    """Resolve CLI ``--variants`` names to :class:`RealismVariant` rows."""
    by_tag = {v.tag: v for v in default_realism_variants()}
    if not tags:
        return list(by_tag.values())
    out: list[RealismVariant] = []
    for t in tags:
        key = str(t).strip()
        if key not in by_tag:
            raise ValueError(f"unknown realism variant {key!r}; choose from {sorted(by_tag)}")
        out.append(by_tag[key])
    return out
