"""Disk cache for session-level EMG preprocessing (train/calibrate speedup)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional, Tuple, Union

import numpy as np

from .emg_config import EmgMode, validate_emg_gowda_fs, validate_emg_wide_fs
from .filters import preprocess_basic, preprocess_streaming

PreprocessMode = Literal["offline", "streaming", "none"]
CACHE_VERSION = 1
CACHE_DIRNAME = "preprocess_cache"
RAW_SIGNALS_NAME = "signals.npy"


@dataclass
class PreprocessCacheReport:
    session_dir: str
    cache_path: str
    preprocess_mode: str
    emg_mode: str
    fs_hz: float
    shape: Tuple[int, int]
    built: bool
    elapsed_s: float
    notes: list[str]

    def to_dict(self) -> dict:
        return {
            "session_dir": self.session_dir,
            "cache_path": self.cache_path,
            "preprocess_mode": self.preprocess_mode,
            "emg_mode": self.emg_mode,
            "fs_hz": float(self.fs_hz),
            "shape": [int(self.shape[0]), int(self.shape[1])],
            "built": bool(self.built),
            "elapsed_s": round(float(self.elapsed_s), 3),
            "notes": list(self.notes),
        }


def _cache_stem(preprocess_mode: str, emg_mode: str, fs_hz: float) -> str:
    return f"{preprocess_mode}_{emg_mode}_{int(fs_hz)}hz"


def cache_paths(
    session_dir: Union[str, Path],
    preprocess_mode: str,
    emg_mode: str,
    fs_hz: float,
) -> Tuple[Path, Path]:
    session_dir = Path(session_dir)
    stem = _cache_stem(preprocess_mode, emg_mode, fs_hz)
    base = session_dir / CACHE_DIRNAME
    return base / f"{stem}.npy", base / f"{stem}.meta.json"


def _raw_signals_path(session_dir: Path) -> Path:
    return session_dir / RAW_SIGNALS_NAME


def _raw_fingerprint(path: Path) -> dict:
    path = Path(path)
    st = path.stat()
    return {
        "path": RAW_SIGNALS_NAME,
        "size_bytes": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def read_cache_meta(meta_path: Path) -> Optional[dict]:
    meta_path = Path(meta_path)
    if not meta_path.is_file():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def is_cache_valid(
    meta_path: Path,
    raw_signals_path: Path,
    *,
    preprocess_mode: str,
    emg_mode: str,
    fs_hz: float,
) -> bool:
    meta = read_cache_meta(meta_path)
    if meta is None:
        return False
    if int(meta.get("cache_version", 0)) != CACHE_VERSION:
        return False
    if str(meta.get("preprocess_mode")) != str(preprocess_mode):
        return False
    if str(meta.get("emg_mode")) != str(emg_mode):
        return False
    if int(meta.get("fs_hz", -1)) != int(fs_hz):
        return False
    if not raw_signals_path.is_file():
        return False
    fp = _raw_fingerprint(raw_signals_path)
    stored = meta.get("raw_signals") or {}
    return (
        str(stored.get("path")) == fp["path"]
        and int(stored.get("size_bytes", -1)) == fp["size_bytes"]
        and int(stored.get("mtime_ns", -1)) == fp["mtime_ns"]
    )


def _run_preprocess(
    signals: np.ndarray,
    *,
    fs_hz: float,
    preprocess_mode: PreprocessMode,
    emg_mode: EmgMode,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> np.ndarray:
    if preprocess_mode == "none":
        return np.asarray(signals, dtype=np.float32)
    if emg_mode == "wide":
        validate_emg_wide_fs(float(fs_hz))
    if emg_mode == "gowda":
        validate_emg_gowda_fs(float(fs_hz))
    if preprocess_mode == "offline":
        if progress_cb:
            progress_cb("preprocess offline (full array)")
        return preprocess_basic(
            signals,
            fs_hz=float(fs_hz),
            mode=emg_mode,
            rectify_signals=False,
            normalize_mode="zscore",
        )
    if preprocess_mode == "streaming":
        if progress_cb:
            progress_cb("preprocess streaming (causal chunks)")

        def _chunk_progress(done: int, total: int) -> None:
            if progress_cb is None:
                return
            if total <= 0:
                return
            pct = min(100.0, 100.0 * done / total)
            if done == 0 or done == total or done % max(1, total // 20) == 0:
                progress_cb(f"streaming preprocess {pct:5.1f}% ({done}/{total} chunks)")

        return preprocess_streaming(
            signals,
            fs_hz=float(fs_hz),
            channels=int(signals.shape[1]),
            rectify_signals=False,
            ema_alpha=0.01,
            mode=emg_mode,
            progress_cb=_chunk_progress,
        )
    raise ValueError(f"unknown preprocess_mode: {preprocess_mode}")


def write_preprocess_cache(
    session_dir: Union[str, Path],
    signals: np.ndarray,
    *,
    preprocess_mode: PreprocessMode,
    emg_mode: EmgMode,
    fs_hz: float,
    raw_signals_path: Optional[Path] = None,
) -> Path:
    """Write preprocessed array + metadata under ``session/preprocess_cache/``."""
    session_dir = Path(session_dir)
    raw_path = Path(raw_signals_path) if raw_signals_path else _raw_signals_path(session_dir)
    npy_path, meta_path = cache_paths(session_dir, preprocess_mode, emg_mode, fs_hz)
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    out = np.asarray(signals, dtype=np.float32)
    np.save(npy_path, out)
    meta = {
        "cache_version": CACHE_VERSION,
        "preprocess_mode": str(preprocess_mode),
        "emg_mode": str(emg_mode),
        "fs_hz": float(fs_hz),
        "shape": [int(out.shape[0]), int(out.shape[1])],
        "dtype": "float32",
        "raw_signals": _raw_fingerprint(raw_path),
        "created_unix": float(time.time()),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return npy_path


def load_cached_signals(
    session_dir: Union[str, Path],
    *,
    preprocess_mode: PreprocessMode,
    emg_mode: EmgMode,
    fs_hz: float,
) -> Optional[np.ndarray]:
    session_dir = Path(session_dir)
    raw_path = _raw_signals_path(session_dir)
    npy_path, meta_path = cache_paths(session_dir, preprocess_mode, emg_mode, fs_hz)
    if not npy_path.is_file():
        return None
    if not is_cache_valid(meta_path, raw_path, preprocess_mode=preprocess_mode, emg_mode=emg_mode, fs_hz=fs_hz):
        return None
    return np.load(npy_path).astype(np.float32, copy=False)


def ensure_preprocessed_signals(
    signals: np.ndarray,
    session_dir: Union[str, Path],
    *,
    preprocess_mode: PreprocessMode,
    emg_mode: EmgMode,
    fs_hz: float,
    use_cache: bool = True,
    rebuild: bool = False,
    show_progress: bool = False,
) -> Tuple[np.ndarray, bool]:
    """Return preprocessed ``(time, channels)`` array; build cache on miss.

    Returns ``(array, cache_hit)``.
    """
    session_dir = Path(session_dir)
    mode = str(preprocess_mode)
    if mode == "none":
        return np.asarray(signals, dtype=np.float32), False

    npy_path, meta_path = cache_paths(session_dir, mode, str(emg_mode), float(fs_hz))
    raw_path = _raw_signals_path(session_dir)

    if use_cache and not rebuild:
        cached = load_cached_signals(
            session_dir,
            preprocess_mode=preprocess_mode,
            emg_mode=emg_mode,
            fs_hz=float(fs_hz),
        )
        if cached is not None:
            return cached, True

    progress_cb: Optional[Callable[[str], None]] = None
    if show_progress:
        progress_cb = lambda msg: print(f"[openalterego] {msg}")

    t0 = time.perf_counter()
    out = _run_preprocess(
        signals,
        fs_hz=float(fs_hz),
        preprocess_mode=preprocess_mode,
        emg_mode=emg_mode,
        progress_cb=progress_cb,
    )
    if use_cache:
        write_preprocess_cache(
            session_dir,
            out,
            preprocess_mode=preprocess_mode,
            emg_mode=emg_mode,
            fs_hz=float(fs_hz),
            raw_signals_path=raw_path if raw_path.is_file() else None,
        )
        if show_progress:
            elapsed = time.perf_counter() - t0
            print(f"[openalterego] wrote preprocess cache: {npy_path} ({elapsed:.1f}s)")
    return out.astype(np.float32, copy=False), False


def build_session_preprocess_cache(
    session_dir: Union[str, Path],
    *,
    preprocess_mode: PreprocessMode = "streaming",
    emg_mode: EmgMode = "wide",
    fs_hz: Optional[float] = None,
    rebuild: bool = False,
    show_progress: bool = True,
) -> PreprocessCacheReport:
    """Build (or refresh) preprocess cache for a session folder."""
    session_dir = Path(session_dir)
    raw_path = _raw_signals_path(session_dir)
    if not raw_path.is_file():
        raise FileNotFoundError(f"missing {raw_path}")
    meta_session: dict = {}
    meta_path = session_dir / "meta.json"
    if meta_path.is_file():
        meta_session = json.loads(meta_path.read_text(encoding="utf-8"))
    fs = float(fs_hz if fs_hz is not None else meta_session.get("fs_hz", 250))

    notes: list[str] = []
    npy_path, _ = cache_paths(session_dir, preprocess_mode, emg_mode, fs)
    t0 = time.perf_counter()
    if (
        not rebuild
        and is_cache_valid(
            cache_paths(session_dir, preprocess_mode, emg_mode, fs)[1],
            raw_path,
            preprocess_mode=preprocess_mode,
            emg_mode=emg_mode,
            fs_hz=fs,
        )
        and npy_path.is_file()
    ):
        arr = np.load(npy_path)
        notes.append("cache hit (already valid)")
        built = False
    else:
        signals = np.load(raw_path)
        arr, _ = ensure_preprocessed_signals(
            signals,
            session_dir,
            preprocess_mode=preprocess_mode,
            emg_mode=emg_mode,
            fs_hz=fs,
            use_cache=True,
            rebuild=True,
            show_progress=show_progress,
        )
        notes.append("cache built")
        built = True

    elapsed = time.perf_counter() - t0
    return PreprocessCacheReport(
        session_dir=str(session_dir),
        cache_path=str(npy_path),
        preprocess_mode=str(preprocess_mode),
        emg_mode=str(emg_mode),
        fs_hz=float(fs),
        shape=(int(arr.shape[0]), int(arr.shape[1])),
        built=built,
        elapsed_s=float(elapsed),
        notes=notes,
    )
