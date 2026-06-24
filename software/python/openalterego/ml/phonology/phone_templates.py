"""Fit per-phone EMG templates from real Gowda sessions (pseudo phone alignment)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ...dsp.filters import preprocess_basic
from ...ml.data_split import gowda_official_train_val_test_indices
from ...ml.datasets.events import load_gowda_events
from ...ml.phonology.gowda_lexicon import build_lexicon
from ...ml.spd.basis import ensure_gowda_spd_basis
from ...ml.spd.features import edge_matrix, sigma_diag_vector, spd_regularize
from ...sim.phonology.align import AlignMode, partition_phones_aligned
from ...sim.phonology.durations import phone_duration_weights
from ...sim.phonology.templates import PhoneTemplate, PhoneTemplateStore, save_phone_templates


@dataclass
class _PhoneAccum:
    rms_sum: np.ndarray
    spd_sum: np.ndarray
    rate_sum: float
    count: int


def _phone_segments_from_event(
    signals: np.ndarray,
    start: int,
    end: int,
    phones: Sequence[str],
    *,
    align_mode: AlignMode,
    seed: int,
) -> List[Tuple[str, np.ndarray]]:
    n = int(end) - int(start)
    if n <= 0 or not phones:
        return []
    seg_lens = partition_phones_aligned(
        n, phones, mode=str(align_mode), seed=int(seed)
    )
    out: List[Tuple[str, np.ndarray]] = []
    cur = int(start)
    for phone, slen in zip(phones, seg_lens):
        s1 = cur + int(slen)
        seg = np.asarray(signals[cur:s1, :], dtype=np.float32)
        if seg.shape[0] >= 8:
            out.append((str(phone), seg))
        cur = s1
    return out


def _pseudo_phone_segments(
    signals: np.ndarray,
    start: int,
    end: int,
    phones: Sequence[str],
    *,
    fs_hz: float,
    seed: int,
    align_mode: AlignMode = "pseudo",
) -> List[Tuple[str, np.ndarray]]:
    del fs_hz  # reserved for future MFA / TTS paths
    return _phone_segments_from_event(
        signals, start, end, phones, align_mode=align_mode, seed=seed
    )


def fit_phone_templates(
    session_dir: Path,
    *,
    split: str = "train",
    max_segments_per_phone: int = 400,
    seed: int = 1337,
    fs_hz: Optional[float] = None,
    align_mode: AlignMode = "pseudo",
) -> PhoneTemplateStore:
    """Aggregate per-phone channel RMS and SPD diag_delta from aligned word events."""
    session_dir = Path(session_dir)
    signals = np.load(session_dir / "signals.npy", mmap_mode="r")
    events = load_gowda_events(session_dir)
    meta = json.loads((session_dir / "meta.json").read_text(encoding="utf-8"))
    fs = float(fs_hz or meta.get("fs_hz", 5000))
    n_ch = int(signals.shape[1])

    trial_ids = events["trial_id"].astype(int).values
    tr_idx, _, _ = gowda_official_train_val_test_indices(trial_ids)
    if str(split) == "train":
        events = events.iloc[tr_idx].reset_index(drop=True)
    elif str(split) != "all":
        raise ValueError(f"unsupported split {split!r}")

    words = sorted(set(str(x) for x in events["label"].astype(str).tolist()))
    lexicon = build_lexicon(words)
    basis = ensure_gowda_spd_basis(session_dir, fs_hz=int(fs), feature_mode="diag_delta")
    feat_dim = int(basis.channels) * 2

    accum: Dict[str, _PhoneAccum] = {}
    rng = np.random.default_rng(int(seed))
    order = np.arange(len(events))
    rng.shuffle(order)

    for row_i in order:
        row = events.iloc[int(row_i)]
        word = str(row["label"])
        phones = tuple(str(p) for p in lexicon.get(word, ()))
        if not phones:
            continue
        s0 = int(row["start_sample"])
        s1 = int(row["end_sample"])
        segs = _pseudo_phone_segments(
            signals,
            s0,
            s1,
            phones,
            fs_hz=fs,
            seed=int(seed) + int(row_i),
            align_mode=str(align_mode),  # type: ignore[arg-type]
        )
        for phone, seg in segs:
            key = str(phone).strip().upper()
            if key not in accum:
                accum[key] = _PhoneAccum(
                    rms_sum=np.zeros((n_ch,), dtype=np.float64),
                    spd_sum=np.zeros((feat_dim,), dtype=np.float64),
                    rate_sum=0.0,
                    count=0,
                )
            if accum[key].count >= int(max_segments_per_phone):
                continue
            proc = preprocess_basic(seg, fs_hz=fs, mode="gowda")
            rms = np.sqrt(np.mean(proc * proc, axis=0))
            edge = spd_regularize(edge_matrix(proc))
            diag = sigma_diag_vector(edge, basis.basis_q).astype(np.float64)
            spd = np.concatenate([diag, np.zeros_like(diag)])
            acc = accum[key]
            acc.rms_sum += rms
            acc.spd_sum += spd
            acc.rate_sum += float(np.mean(rms))
            acc.count += 1

    phones_out: Dict[str, PhoneTemplate] = {}
    global_rate = float(np.mean([a.rate_sum / max(a.count, 1) for a in accum.values()])) if accum else 1.0
    for phone, acc in accum.items():
        if acc.count < 1:
            continue
        rms_mean = acc.rms_sum / float(acc.count)
        spd_mean = acc.spd_sum / float(acc.count)
        rate = float(acc.rate_sum / float(acc.count))
        phones_out[phone] = PhoneTemplate(
            phone=phone,
            channel_rms=rms_mean.astype(np.float64),
            spd_diag_delta=spd_mean.astype(np.float64),
            rate_scale=float(rate / (global_rate + 1e-9)),
            duration_weight=float(phone_duration_weights([phone])[0]),
            n_segments=int(acc.count),
        )

    store = PhoneTemplateStore(
        phones=phones_out,
        n_channels=n_ch,
        feature_dim=feat_dim,
        meta={
            "session_dir": str(session_dir),
            "split": str(split),
            "n_events": int(len(events)),
            "max_segments_per_phone": int(max_segments_per_phone),
            "seed": int(seed),
            "fs_hz": fs,
            "alignment": str(align_mode),
        },
    )
    return store


def fit_and_save_phone_templates(
    session_dir: Path,
    out_path: Path,
    **kwargs: Any,
) -> PhoneTemplateStore:
    store = fit_phone_templates(session_dir, **kwargs)
    save_phone_templates(store, out_path)
    return store
