"""Fit and cache subject/session SPD eigenbasis Q from training windows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .features import (
    GOWDA_SPD_ETA,
    GOWDA_SPD_STEP_MS,
    GOWDA_SPD_WINDOW_MS,
    eigenbasis_from_frechet,
    feature_dim_for_channels,
    log_euclidean_mean,
    sliding_edge_matrices,
)

SPD_CACHE_DIR = "spd_cache"


@dataclass
class SPDBasis:
    """Fixed eigenbasis Q and metadata for σ(τ) features."""

    basis_q: np.ndarray
    channels: int
    fs_hz: int
    window_ms: int
    step_ms: int
    eta: float
    n_train_matrices: int
    feature_dim: int
    use_upper_tri: bool = False
    feature_mode: str = "full"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channels": int(self.channels),
            "fs_hz": int(self.fs_hz),
            "window_ms": int(self.window_ms),
            "step_ms": int(self.step_ms),
            "eta": float(self.eta),
            "n_train_matrices": int(self.n_train_matrices),
            "feature_dim": int(self.feature_dim),
            "use_upper_tri": bool(self.use_upper_tri),
            "feature_mode": str(self.feature_mode),
        }


def _preprocess_segment(seg: np.ndarray, *, fs_hz: int, emg_mode: str) -> np.ndarray:
    from ...dsp.filters import preprocess_basic

    return preprocess_basic(
        np.asarray(seg, dtype=np.float32),
        fs_hz=int(fs_hz),
        mode=emg_mode,  # type: ignore[arg-type]
        rectify_signals=False,
        normalize_mode="zscore",
    )


def collect_training_edge_matrices(
    signals: np.ndarray,
    events: pd.DataFrame,
    *,
    fs_hz: int,
    emg_mode: str = "gowda",
    window_ms: int = GOWDA_SPD_WINDOW_MS,
    step_ms: int = GOWDA_SPD_STEP_MS,
    eta: float = GOWDA_SPD_ETA,
    max_matrices: Optional[int] = 8000,
    seed: int = 1337,
) -> List[np.ndarray]:
    """Collect ℰ(τ) from training events (optionally subsampled)."""
    matrices: List[np.ndarray] = []
    rng = np.random.default_rng(int(seed))
    rows = list(events.iterrows())
    if max_matrices is not None and len(rows) > 0:
        rng.shuffle(rows)

    for _, row in rows:
        s, e = int(row["start_sample"]), int(row["end_sample"])
        seg = _preprocess_segment(signals[s:e, :], fs_hz=int(fs_hz), emg_mode=str(emg_mode))
        mats = sliding_edge_matrices(
            seg,
            fs_hz=int(fs_hz),
            window_ms=int(window_ms),
            step_ms=int(step_ms),
            eta=float(eta),
        )
        for m in mats:
            matrices.append(m)
            if max_matrices is not None and len(matrices) >= int(max_matrices):
                return matrices
    return matrices


def fit_spd_basis(
    matrices: List[np.ndarray],
    *,
    channels: int,
    fs_hz: int,
    window_ms: int = GOWDA_SPD_WINDOW_MS,
    step_ms: int = GOWDA_SPD_STEP_MS,
    eta: float = GOWDA_SPD_ETA,
    use_upper_tri: bool = False,
    feature_mode: str = "full",
) -> SPDBasis:
    if not matrices:
        raise ValueError("no edge matrices to fit SPD basis")
    frechet = log_euclidean_mean(matrices, eta=float(eta))
    q = eigenbasis_from_frechet(frechet)
    c = int(channels)
    mode = "upper_tri" if use_upper_tri and str(feature_mode) == "full" else str(feature_mode)
    return SPDBasis(
        basis_q=q.astype(np.float64, copy=False),
        channels=c,
        fs_hz=int(fs_hz),
        window_ms=int(window_ms),
        step_ms=int(step_ms),
        eta=float(eta),
        n_train_matrices=int(len(matrices)),
        feature_dim=feature_dim_for_channels(c, feature_mode=mode),
        use_upper_tri=mode == "upper_tri",
        feature_mode=mode,
    )


def fit_spd_basis_from_events(
    signals: np.ndarray,
    events: pd.DataFrame,
    *,
    fs_hz: int,
    emg_mode: str = "gowda",
    window_ms: int = GOWDA_SPD_WINDOW_MS,
    step_ms: int = GOWDA_SPD_STEP_MS,
    eta: float = GOWDA_SPD_ETA,
    max_matrices: Optional[int] = 8000,
    seed: int = 1337,
    use_upper_tri: bool = False,
) -> SPDBasis:
    matrices = collect_training_edge_matrices(
        signals,
        events,
        fs_hz=int(fs_hz),
        emg_mode=str(emg_mode),
        window_ms=int(window_ms),
        step_ms=int(step_ms),
        eta=float(eta),
        max_matrices=max_matrices,
        seed=int(seed),
    )
    ch = int(signals.shape[1])
    return fit_spd_basis(
        matrices,
        channels=ch,
        fs_hz=int(fs_hz),
        window_ms=int(window_ms),
        step_ms=int(step_ms),
        eta=float(eta),
        use_upper_tri=use_upper_tri,
    )


def basis_cache_stem(
    *,
    emg_mode: str,
    fs_hz: int,
    window_ms: int,
    step_ms: int,
    eta: float,
    split_tag: str,
    use_upper_tri: bool = False,
) -> str:
    eta_tag = str(eta).replace(".", "p")
    tri = "_utri" if use_upper_tri else ""
    return f"basis_{emg_mode}_{int(fs_hz)}hz_w{int(window_ms)}_s{int(step_ms)}_e{eta_tag}_{split_tag}{tri}"


def basis_cache_paths(session_dir: Union[str, Path], stem: str) -> tuple[Path, Path]:
    base = Path(session_dir) / SPD_CACHE_DIR
    return base / f"{stem}.npz", base / f"{stem}.meta.json"


def save_spd_basis(session_dir: Union[str, Path], stem: str, basis: SPDBasis) -> Path:
    npz_path, meta_path = basis_cache_paths(session_dir, stem)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, basis_q=basis.basis_q.astype(np.float64, copy=False))
    meta = {"stem": stem, **basis.to_dict()}
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return npz_path


def load_spd_basis(session_dir: Union[str, Path], stem: str) -> SPDBasis:
    npz_path, meta_path = basis_cache_paths(session_dir, stem)
    if not npz_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"SPD basis cache missing: {npz_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    q = np.load(npz_path)["basis_q"]
    return SPDBasis(
        basis_q=np.asarray(q, dtype=np.float64),
        channels=int(meta["channels"]),
        fs_hz=int(meta["fs_hz"]),
        window_ms=int(meta["window_ms"]),
        step_ms=int(meta["step_ms"]),
        eta=float(meta["eta"]),
        n_train_matrices=int(meta["n_train_matrices"]),
        feature_dim=int(meta["feature_dim"]),
        use_upper_tri=bool(meta.get("use_upper_tri", False)),
        feature_mode=str(meta.get("feature_mode", "upper_tri" if meta.get("use_upper_tri") else "full")),
    )


def ensure_gowda_spd_basis(
    session_dir: Union[str, Path],
    *,
    basis_session_dir: Optional[Union[str, Path]] = None,
    fs_hz: int,
    emg_mode: str = "gowda",
    window_ms: int = GOWDA_SPD_WINDOW_MS,
    step_ms: int = GOWDA_SPD_STEP_MS,
    eta: float = GOWDA_SPD_ETA,
    max_matrices: Optional[int] = 8000,
    seed: int = 1337,
    split_tag: str = "gowda_train",
    use_upper_tri: bool = False,
    feature_mode: str = "full",
) -> SPDBasis:
    """Load cached Q or fit on Gowda sentence-train split only.

    When ``basis_session_dir`` is set (e.g. real OSF session), load/fit the eigenbasis
    from that directory while ``session_dir`` may point at sim data for training.
    """
    session_dir = Path(session_dir)
    cache_dir = Path(basis_session_dir) if basis_session_dir is not None else session_dir
    mode = "upper_tri" if use_upper_tri and str(feature_mode) == "full" else str(feature_mode)
    stem = basis_cache_stem(
        emg_mode=str(emg_mode),
        fs_hz=int(fs_hz),
        window_ms=int(window_ms),
        step_ms=int(step_ms),
        eta=float(eta),
        split_tag=str(split_tag),
        use_upper_tri=False,
    )
    try:
        basis = load_spd_basis(cache_dir, stem)
    except FileNotFoundError:
        from ..data_split import resolve_gowda_train_val_test_indices
        from ..datasets.events import load_gowda_events, read_split_mode

        signals = np.load(cache_dir / "signals.npy", mmap_mode="r")
        events = load_gowda_events(cache_dir)
        if "trial_id" in events.columns:
            tr_idx, _, _ = resolve_gowda_train_val_test_indices(
                events["trial_id"].astype(int).values,
                split_mode=read_split_mode(cache_dir),
            )
            tr_events = events.iloc[tr_idx].reset_index(drop=True)
        else:
            tr_events = events

        basis = fit_spd_basis_from_events(
            np.asarray(signals, dtype=np.float32),
            tr_events,
            fs_hz=int(fs_hz),
            emg_mode=str(emg_mode),
            window_ms=int(window_ms),
            step_ms=int(step_ms),
            eta=float(eta),
            max_matrices=max_matrices,
            seed=int(seed),
            use_upper_tri=(mode == "upper_tri"),
        )
        save_spd_basis(cache_dir, stem, basis)

    fd = feature_dim_for_channels(int(basis.channels), feature_mode=mode)
    return SPDBasis(
        basis_q=basis.basis_q,
        channels=int(basis.channels),
        fs_hz=int(basis.fs_hz),
        window_ms=int(basis.window_ms),
        step_ms=int(basis.step_ms),
        eta=float(basis.eta),
        n_train_matrices=int(basis.n_train_matrices),
        feature_dim=int(fd),
        use_upper_tri=mode == "upper_tri",
        feature_mode=mode,
    )
