"""Fine-tune an open-speech CTC checkpoint on a user session."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from ..ml.ctc.train import train_gowda_ctc


def fine_tune_ctc_adapter(
    session_dir: Path | str,
    *,
    init_checkpoint: Path | str,
    out_checkpoint: Path | str,
    basis_session_dir: Optional[Path | str] = None,
    epochs: int = 10,
    lr: float = 5e-4,
    seed: int = 1337,
    device_preferred: str = "auto",
    feature_mode: str = "diag_delta",
) -> Dict[str, Any]:
    """Short adapter fine-tune on user EMG (official train split)."""
    session_dir = Path(session_dir)
    out_checkpoint = Path(out_checkpoint)
    out_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    basis_dir = Path(basis_session_dir) if basis_session_dir is not None else session_dir

    row = train_gowda_ctc(
        session_dir,
        fs_hz=5000,
        feature_type="spd",
        feature_mode=str(feature_mode),
        hidden=256,
        num_layers=3,
        dropout=0.2,
        epochs=int(epochs),
        batch_size=32,
        lr=float(lr),
        warmup_epochs=2,
        seed=int(seed),
        device_preferred=device_preferred,
        val_decode_mode="greedy",
        eval_test=False,
        save_path=out_checkpoint,
        basis_session_dir=basis_dir,
        init_checkpoint=Path(init_checkpoint),
    )
    return row


def install_checkpoint_for_user(
    src_checkpoint: Path | str,
    user_dir: Path | str,
    *,
    filename: str = "ctc_open_speech.pt",
) -> Path:
    """Copy a CTC checkpoint into the user directory."""
    user_dir = Path(user_dir)
    user_dir.mkdir(parents=True, exist_ok=True)
    dst = user_dir / filename
    shutil.copy2(Path(src_checkpoint), dst)
    return dst
