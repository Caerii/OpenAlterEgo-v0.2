"""Disk cache for training segment tensors (N, channels, time) + labels."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..dsp.preprocess_cache import cache_paths as preprocess_cache_paths

CACHE_VERSION = 1
SEGMENTS_DIRNAME = "segments"


@dataclass
class SegmentCacheReport:
    cache_path: str
    n_segments: int
    shape: Tuple[int, int, int]
    built: bool
    elapsed_s: float

    def to_dict(self) -> dict:
        return {
            "cache_path": self.cache_path,
            "n_segments": int(self.n_segments),
            "shape": [int(x) for x in self.shape],
            "built": bool(self.built),
            "elapsed_s": round(float(self.elapsed_s), 3),
        }


def _channel_tag(channel_indices: Optional[List[int]]) -> str:
    if not channel_indices:
        return "all"
    return "c" + "-".join(str(int(i)) for i in channel_indices)


def _events_fingerprint(events: pd.DataFrame) -> str:
    cols = [c for c in ("start_sample", "end_sample", "label") if c in events.columns]
    payload = events[cols].to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def segment_cache_stem(
    *,
    preprocess_mode: str,
    emg_mode: str,
    fs_hz: int,
    segment_ms: int,
    seed: int,
    split_tag: str,
    channel_indices: Optional[List[int]],
    events: pd.DataFrame,
    per_event_preprocess: bool = False,
) -> str:
    ev_fp = _events_fingerprint(events)
    pe = "_pe1" if per_event_preprocess else ""
    return (
        f"v{CACHE_VERSION}_{preprocess_mode}_{emg_mode}_{int(fs_hz)}hz"
        f"_ms{int(segment_ms)}_seed{int(seed)}_{split_tag}"
        f"_{_channel_tag(channel_indices)}_{ev_fp}{pe}"
    )


def segment_cache_paths(session_dir: Union[str, Path], stem: str) -> Tuple[Path, Path]:
    base = Path(session_dir) / SEGMENTS_DIRNAME
    return base / f"{stem}.npz", base / f"{stem}.meta.json"


def _preprocess_meta_path(session_dir: Path, preprocess_mode: str, emg_mode: str, fs_hz: float) -> Path:
    return preprocess_cache_paths(session_dir, preprocess_mode, emg_mode, fs_hz)[1]


def is_segment_cache_valid(
    meta_path: Path,
    *,
    preprocess_meta_path: Path,
    preprocess_mode: str,
    emg_mode: str,
    fs_hz: int,
    segment_ms: int,
    seed: int,
    split_tag: str,
    channel_indices: Optional[List[int]],
    events: pd.DataFrame,
) -> bool:
    if not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if int(meta.get("cache_version", 0)) != CACHE_VERSION:
        return False
    if str(meta.get("preprocess_mode")) != str(preprocess_mode):
        return False
    if str(meta.get("emg_mode")) != str(emg_mode):
        return False
    if int(meta.get("fs_hz", -1)) != int(fs_hz):
        return False
    if int(meta.get("segment_ms", -1)) != int(segment_ms):
        return False
    if int(meta.get("seed", -1)) != int(seed):
        return False
    if str(meta.get("split_tag")) != str(split_tag):
        return False
    if str(meta.get("channel_tag")) != _channel_tag(channel_indices):
        return False
    if str(meta.get("events_fingerprint")) != _events_fingerprint(events):
        return False
    if preprocess_meta_path.is_file():
        pp = json.loads(preprocess_meta_path.read_text(encoding="utf-8"))
        stored = meta.get("preprocess_cache") or {}
        if int(stored.get("cache_version", -1)) != int(pp.get("cache_version", -2)):
            return False
        raw = pp.get("raw_signals") or {}
        if stored.get("raw_signals") != raw:
            return False
    return True


def build_segment_arrays(
    signals: np.ndarray,
    events: pd.DataFrame,
    label_to_id: Dict[str, int],
    *,
    fs_hz: int,
    segment_ms: int,
    seed: int,
    channel_indices: Optional[List[int]] = None,
    per_event_preprocess: bool = False,
    preprocess_emg_mode: str = "standard",
) -> Tuple[np.ndarray, np.ndarray]:
    """Materialize ``(N, C, T)`` float32 segments and ``(N,)`` int64 labels."""
    from ..dsp.filters import preprocess_basic

    rng = np.random.default_rng(int(seed))
    segment_samples = max(8, int(fs_hz) * int(segment_ms) // 1000)
    ch_idx = list(channel_indices) if channel_indices is not None else None
    n_ch = len(ch_idx) if ch_idx is not None else int(signals.shape[1])

    xs: List[np.ndarray] = []
    ys: List[int] = []
    for _, row in events.iterrows():
        s, e = int(row["start_sample"]), int(row["end_sample"])
        lab = str(row["label"])
        if lab not in label_to_id:
            continue
        seg = np.asarray(signals[s:e, :], dtype=np.float32)
        if per_event_preprocess:
            seg = preprocess_basic(
                seg,
                fs_hz=int(fs_hz),
                mode=preprocess_emg_mode,  # type: ignore[arg-type]
                rectify_signals=False,
                normalize_mode="zscore",
            )
        if seg.shape[0] < 8:
            continue
        x = seg.T
        if ch_idx is not None:
            x = x[ch_idx, :]
        ch, t = x.shape
        if t == segment_samples:
            out = x.astype(np.float32, copy=False)
        elif t > segment_samples:
            start = int(rng.integers(0, t - segment_samples + 1))
            out = x[:, start : start + segment_samples].astype(np.float32, copy=False)
        else:
            out = np.zeros((ch, segment_samples), dtype=np.float32)
            out[:, :t] = x.astype(np.float32, copy=False)
        xs.append(out)
        ys.append(int(label_to_id[lab]))

    if not xs:
        return np.zeros((0, n_ch, segment_samples), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    X = np.stack(xs, axis=0).astype(np.float32, copy=False)
    y = np.asarray(ys, dtype=np.int64)
    return X, y


def write_segment_cache(
    session_dir: Union[str, Path],
    stem: str,
    X: np.ndarray,
    y: np.ndarray,
    *,
    preprocess_mode: str,
    emg_mode: str,
    fs_hz: int,
    segment_ms: int,
    seed: int,
    split_tag: str,
    channel_indices: Optional[List[int]],
    events: pd.DataFrame,
) -> Path:
    session_dir = Path(session_dir)
    npz_path, meta_path = segment_cache_paths(session_dir, stem)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, X=X.astype(np.float32, copy=False), y=y.astype(np.int64, copy=False))
    pp_meta: dict = {}
    pp_path = _preprocess_meta_path(session_dir, preprocess_mode, emg_mode, float(fs_hz))
    if pp_path.is_file():
        try:
            pp_meta = json.loads(pp_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pp_meta = {}
    meta = {
        "cache_version": CACHE_VERSION,
        "preprocess_mode": str(preprocess_mode),
        "emg_mode": str(emg_mode),
        "fs_hz": int(fs_hz),
        "segment_ms": int(segment_ms),
        "seed": int(seed),
        "split_tag": str(split_tag),
        "channel_tag": _channel_tag(channel_indices),
        "events_fingerprint": _events_fingerprint(events),
        "shape": [int(X.shape[0]), int(X.shape[1]), int(X.shape[2])],
        "preprocess_cache": {
            "cache_version": pp_meta.get("cache_version"),
            "raw_signals": pp_meta.get("raw_signals"),
        },
        "created_unix": float(time.time()),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return npz_path


def load_segment_cache(
    session_dir: Union[str, Path],
    stem: str,
    *,
    preprocess_mode: str,
    emg_mode: str,
    fs_hz: int,
    segment_ms: int,
    seed: int,
    split_tag: str,
    channel_indices: Optional[List[int]],
    events: pd.DataFrame,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    session_dir = Path(session_dir)
    npz_path, meta_path = segment_cache_paths(session_dir, stem)
    if not npz_path.is_file():
        return None
    pp_path = _preprocess_meta_path(session_dir, preprocess_mode, emg_mode, float(fs_hz))
    if not is_segment_cache_valid(
        meta_path,
        preprocess_meta_path=pp_path,
        preprocess_mode=preprocess_mode,
        emg_mode=emg_mode,
        fs_hz=int(fs_hz),
        segment_ms=int(segment_ms),
        seed=int(seed),
        split_tag=split_tag,
        channel_indices=channel_indices,
        events=events,
    ):
        return None
    data = np.load(npz_path)
    return data["X"].astype(np.float32, copy=False), data["y"].astype(np.int64, copy=False)


def ensure_segment_arrays(
    signals: np.ndarray,
    events: pd.DataFrame,
    label_to_id: Dict[str, int],
    session_dir: Union[str, Path],
    *,
    preprocess_mode: str,
    emg_mode: str,
    fs_hz: int,
    segment_ms: int,
    seed: int,
    split_tag: str,
    channel_indices: Optional[List[int]] = None,
    use_cache: bool = True,
    rebuild: bool = False,
    show_progress: bool = False,
    per_event_preprocess: bool = False,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Return ``(X, y, cache_hit)``."""
    session_dir = Path(session_dir)
    stem = segment_cache_stem(
        preprocess_mode=preprocess_mode,
        emg_mode=emg_mode,
        fs_hz=int(fs_hz),
        segment_ms=int(segment_ms),
        seed=int(seed),
        split_tag=split_tag,
        channel_indices=channel_indices,
        events=events,
        per_event_preprocess=bool(per_event_preprocess),
    )
    if use_cache and not rebuild:
        loaded = load_segment_cache(
            session_dir,
            stem,
            preprocess_mode=preprocess_mode,
            emg_mode=emg_mode,
            fs_hz=int(fs_hz),
            segment_ms=int(segment_ms),
            seed=int(seed),
            split_tag=split_tag,
            channel_indices=channel_indices,
            events=events,
        )
        if loaded is not None:
            return loaded[0], loaded[1], True

    t0 = time.perf_counter()
    X, y = build_segment_arrays(
        signals,
        events,
        label_to_id,
        fs_hz=int(fs_hz),
        segment_ms=int(segment_ms),
        seed=int(seed),
        channel_indices=channel_indices,
        per_event_preprocess=bool(per_event_preprocess),
        preprocess_emg_mode=str(emg_mode),
    )
    if use_cache:
        write_segment_cache(
            session_dir,
            stem,
            X,
            y,
            preprocess_mode=preprocess_mode,
            emg_mode=emg_mode,
            fs_hz=int(fs_hz),
            segment_ms=int(segment_ms),
            seed=int(seed),
            split_tag=split_tag,
            channel_indices=channel_indices,
            events=events,
        )
        if show_progress:
            npz_path, _ = segment_cache_paths(session_dir, stem)
            elapsed = time.perf_counter() - t0
            print(f"[openalterego] wrote segment cache: {npz_path} ({elapsed:.1f}s, n={len(y)})")
    return X, y, False
