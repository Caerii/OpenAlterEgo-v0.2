"""Open-speech user onboarding: per-user SPD basis + optional CTC adapter."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional

from ..ml.spd.basis_store import (
    copy_session_spd_cache_to_user,
    fit_user_basis_from_session,
    user_basis_exists,
)
from .adapter_train import fine_tune_ctc_adapter, install_checkpoint_for_user
from .manager import UserManager
from .profile import UserProfile


def onboard_open_speech(
    user_id: str,
    session_dir: Path | str,
    *,
    users_dir: Path | str,
    base_checkpoint: Optional[Path | str] = None,
    fit_basis: bool = True,
    adapter_epochs: int = 0,
    feature_mode: str = "diag_delta",
    seed: int = 1337,
    device_preferred: str = "auto",
) -> Dict[str, Any]:
    """Create/update user with per-user SPD Q and optional CTC adapter fine-tune.

    Parameters
    ----------
    user_id:
        User identifier (profile created if missing).
    session_dir:
        Collected or imported Gowda-shaped session (signals.npy + events.csv).
    base_checkpoint:
        Starting CTC weights (e.g. global Phase 6 checkpoint). Required if ``adapter_epochs > 0``.
    adapter_epochs:
        If > 0, fine-tune ``base_checkpoint`` on ``session_dir`` train split.
    """
    session_dir = Path(session_dir)
    users_dir = Path(users_dir)
    if not (session_dir / "signals.npy").is_file():
        raise FileNotFoundError(f"missing signals.npy in {session_dir}")
    if not (session_dir / "events.csv").is_file():
        raise FileNotFoundError(f"missing events.csv in {session_dir}")

    mgr = UserManager(users_dir)
    profile = mgr.load_profile(user_id)
    if profile is None:
        profile = UserProfile(user_id=user_id)
        mgr.save_profile(profile)

    user_dir = mgr.get_user_dir(user_id)
    report: Dict[str, Any] = {
        "user_id": user_id,
        "session_dir": str(session_dir),
        "user_dir": str(user_dir),
    }

    if fit_basis:
        copied = copy_session_spd_cache_to_user(session_dir, user_dir)
        if copied is None or not user_basis_exists(user_dir):
            basis = fit_user_basis_from_session(
                session_dir,
                user_dir,
                feature_mode=str(feature_mode),
                seed=int(seed),
            )
            report["basis"] = {"fitted": True, "n_train_matrices": int(basis.n_train_matrices)}
        else:
            report["basis"] = {"fitted": False, "copied_from_session_cache": True}
    else:
        report["basis"] = {"skipped": True}

    ckpt_path: Optional[Path] = None
    if int(adapter_epochs) > 0:
        if base_checkpoint is None:
            raise ValueError("base_checkpoint required when adapter_epochs > 0")
        ckpt_path = user_dir / "ctc_open_speech.pt"
        train_row = fine_tune_ctc_adapter(
            session_dir,
            init_checkpoint=Path(base_checkpoint),
            out_checkpoint=ckpt_path,
            basis_session_dir=session_dir,
            epochs=int(adapter_epochs),
            seed=int(seed),
            device_preferred=device_preferred,
            feature_mode=str(feature_mode),
        )
        report["adapter"] = {k: v for k, v in train_row.items() if k != "state_dict"}
    elif base_checkpoint is not None:
        ckpt_path = install_checkpoint_for_user(base_checkpoint, user_dir)

    if ckpt_path is not None:
        profile = replace(
            profile,
            model_path=ckpt_path,
            calibration_date=time.time(),
        )
        mgr.save_profile(profile)
        report["checkpoint"] = str(ckpt_path)

    report["profile"] = profile.to_dict()
    return report
