"""Per-user SPD eigenbasis persistence (Gowda change-of-basis)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional, Union

import numpy as np

from ..data_split import resolve_gowda_train_val_test_indices
from ..datasets.events import load_gowda_events, read_split_mode
from .basis import SPDBasis, fit_spd_basis_from_events, load_spd_basis, save_spd_basis, SPD_CACHE_DIR

USER_BASIS_STEM = "user_gowda_basis"


def user_basis_dir(user_dir: Path | str) -> Path:
    return Path(user_dir) / "spd_basis"


def fit_user_basis_from_session(
    session_dir: Path | str,
    user_dir: Path | str,
    *,
    fs_hz: int = 5000,
    emg_mode: str = "gowda",
    feature_mode: str = "diag_delta",
    seed: int = 1337,
    max_matrices: Optional[int] = 8000,
) -> SPDBasis:
    """Fit Q on session train split and save under ``user_dir/spd_basis/``."""
    session_dir = Path(session_dir)
    user_dir = Path(user_dir)
    user_dir.mkdir(parents=True, exist_ok=True)

    signals = np.load(session_dir / "signals.npy", mmap_mode="r")
    events = load_gowda_events(session_dir)
    if "trial_id" in events.columns and len(events):
        tr_idx, _, _ = resolve_gowda_train_val_test_indices(
            events["trial_id"].astype(int).values,
            split_mode=read_split_mode(session_dir),
        )
        tr_events = events.iloc[tr_idx].reset_index(drop=True)
    else:
        tr_events = events

    basis = fit_spd_basis_from_events(
        np.asarray(signals, dtype=np.float32),
        tr_events,
        fs_hz=int(fs_hz),
        emg_mode=str(emg_mode),
        max_matrices=max_matrices,
        seed=int(seed),
    )
    basis = SPDBasis(
        basis_q=basis.basis_q,
        channels=int(basis.channels),
        fs_hz=int(basis.fs_hz),
        window_ms=int(basis.window_ms),
        step_ms=int(basis.step_ms),
        eta=float(basis.eta),
        n_train_matrices=int(basis.n_train_matrices),
        feature_dim=int(basis.feature_dim),
        use_upper_tri=False,
        feature_mode=str(feature_mode),
    )
    out_dir = user_basis_dir(user_dir)
    save_spd_basis(out_dir, USER_BASIS_STEM, basis)
    meta = {
        "source_session": str(session_dir.resolve()),
        "stem": USER_BASIS_STEM,
        "feature_mode": str(feature_mode),
        **basis.to_dict(),
    }
    _, meta_path = _user_basis_paths(user_dir)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return basis


def _user_basis_paths(user_dir: Path | str) -> tuple[Path, Path]:
    base = user_basis_dir(user_dir) / SPD_CACHE_DIR
    return base / f"{USER_BASIS_STEM}.npz", base / f"{USER_BASIS_STEM}.meta.json"


def load_user_basis(user_dir: Path | str) -> SPDBasis:
    user_dir = Path(user_dir)
    return load_spd_basis(user_basis_dir(user_dir), USER_BASIS_STEM)


def user_basis_exists(user_dir: Path | str) -> bool:
    npz, meta = _user_basis_paths(user_dir)
    return npz.is_file() and meta.is_file()


def copy_session_spd_cache_to_user(session_dir: Path | str, user_dir: Path | str) -> Optional[Path]:
    """Copy session ``spd_cache`` into user dir if present (reuse OSF fit)."""
    src = Path(session_dir) / "spd_cache"
    if not src.is_dir():
        return None
    dst = user_basis_dir(user_dir)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst
