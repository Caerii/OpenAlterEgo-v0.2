"""Disk cache for σ(τ) sequence tensors per event split."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .basis import SPDBasis, basis_cache_stem
from .sequences import build_spd_segment_sequences

CACHE_VERSION = 1
SEQ_DIR = "spd_sequences"


def _events_fp(events: pd.DataFrame) -> str:
    cols = [c for c in ("start_sample", "end_sample", "label") if c in events.columns]
    return hashlib.sha256(events[cols].to_csv(index=False).encode()).hexdigest()[:16]


def sequence_cache_stem(
    *,
    basis_stem: str,
    split_tag: str,
    events: pd.DataFrame,
    emg_mode: str,
    feature_mode: str = "full",
) -> str:
    fm = str(feature_mode).replace("_", "")
    return f"v{CACHE_VERSION}_{basis_stem}_{fm}_{split_tag}_{emg_mode}_{_events_fp(events)}"


def sequence_cache_paths(session_dir: Union[str, Path], stem: str) -> Tuple[Path, Path]:
    base = Path(session_dir) / SEQ_DIR
    return base / f"{stem}.npz", base / f"{stem}.meta.json"


def load_or_build_spd_sequences(
    signals: np.ndarray,
    events: pd.DataFrame,
    label_to_id: dict,
    basis: SPDBasis,
    session_dir: Union[str, Path],
    *,
    split_tag: str,
    fs_hz: int,
    emg_mode: str = "gowda",
    per_event_preprocess: bool = True,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Return σ sequences, building disk cache on first access."""
    session_dir = Path(session_dir)
    bstem = basis_cache_stem(
        emg_mode=str(emg_mode),
        fs_hz=int(basis.fs_hz),
        window_ms=int(basis.window_ms),
        step_ms=int(basis.step_ms),
        eta=float(basis.eta),
        split_tag="gowda_train",
        use_upper_tri=bool(basis.use_upper_tri),
    )
    stem = sequence_cache_stem(
        basis_stem=bstem,
        split_tag=str(split_tag),
        events=events,
        emg_mode=str(emg_mode),
        feature_mode=str(basis.feature_mode),
    )
    npz_path, meta_path = sequence_cache_paths(session_dir, stem)
    if npz_path.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if int(meta.get("feature_dim", -1)) == int(basis.feature_dim):
            data = np.load(npz_path, allow_pickle=True)
            seqs = [np.asarray(x, dtype=np.float32) for x in data["seqs"]]
            if seqs and int(seqs[0].shape[1]) == int(basis.feature_dim):
                y = np.asarray(data["y"], dtype=np.int64)
                return seqs, y

    seqs, y = build_spd_segment_sequences(
        signals,
        events,
        label_to_id,
        basis,
        fs_hz=int(fs_hz),
        emg_mode=str(emg_mode),
        per_event_preprocess=per_event_preprocess,
    )
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, seqs=np.array(seqs, dtype=object), y=y)
    meta_path.write_text(
        json.dumps(
            {
                "cache_version": CACHE_VERSION,
                "stem": stem,
                "n_segments": len(seqs),
                "feature_dim": int(basis.feature_dim),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return seqs, y
