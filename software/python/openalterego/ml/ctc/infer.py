"""Load SPD+CTC checkpoints for inference and offline decode."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from ..device import resolve_device
from ..phonology.gowda_lexicon import PHONEME_ALPHABET
from ..spd.basis import SPDBasis, ensure_gowda_spd_basis
from .eval import build_ctc_model_from_checkpoint


@dataclass
class LoadedCTCModel:
    model: torch.nn.Module
    device: torch.device
    checkpoint_path: Path
    feature_type: str
    feature_dim: int
    feature_mode: str
    fs_hz: int
    channels: int
    n_phonemes: int
    hidden: int
    num_layers: int
    spd_basis: Optional[SPDBasis]
    basis_q: Optional[np.ndarray]
    emg_mode: str
    seed: int

    @property
    def uses_spd(self) -> bool:
        return str(self.feature_type) == "spd"


def load_ctc_model(
    path: Path | str,
    *,
    device_preferred: str = "auto",
    session_dir: Optional[Path | str] = None,
) -> LoadedCTCModel:
    """Load a Gowda SPD+CTC checkpoint for inference."""
    ckpt_path = Path(path)
    device = resolve_device(device_preferred)
    ckpt: dict[str, Any] = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_ctc_model_from_checkpoint(ckpt, device)

    ft = str(ckpt.get("feature_type", "raw"))
    spd_basis: Optional[SPDBasis] = None
    basis_q: Optional[np.ndarray] = None
    sess = Path(session_dir) if session_dir is not None else ckpt_path.parent.parent

    if ft == "spd":
        spd_basis = ensure_gowda_spd_basis(
            sess,
            fs_hz=int(ckpt.get("fs", 5000)),
            emg_mode=str(ckpt.get("emg_mode", "gowda")),
            seed=int(ckpt.get("seed", 1337)),
            use_upper_tri=bool(ckpt.get("use_upper_tri", False)),
            feature_mode=str(ckpt.get("feature_mode", "diag_delta")),
        )
        basis_q = np.asarray(spd_basis.basis_q, dtype=np.float64)

    return LoadedCTCModel(
        model=model,
        device=device,
        checkpoint_path=ckpt_path,
        feature_type=ft,
        feature_dim=int(ckpt.get("feature_dim", 0)),
        feature_mode=str(ckpt.get("feature_mode", "full")),
        fs_hz=int(ckpt.get("fs", 5000)),
        channels=int(ckpt.get("channels", 31)),
        n_phonemes=int(ckpt.get("n_phonemes", len(PHONEME_ALPHABET))),
        hidden=int(ckpt.get("hidden", 256)),
        num_layers=int(ckpt.get("num_layers", 2)),
        spd_basis=spd_basis,
        basis_q=basis_q,
        emg_mode=str(ckpt.get("emg_mode", "gowda")),
        seed=int(ckpt.get("seed", 1337)),
    )


def forward_log_probs(
    loaded: LoadedCTCModel,
    sigma_sequence: np.ndarray,
) -> np.ndarray:
    """Run CTC encoder; return log-probs ``(T, C)``."""
    x = torch.from_numpy(np.asarray(sigma_sequence, dtype=np.float32)).unsqueeze(0).to(loaded.device)
    loaded.model.eval()
    with torch.no_grad():
        logits = loaded.model(x)
        lp = logits[0].detach().cpu().numpy()
    lp = lp - np.max(lp, axis=-1, keepdims=True)
    return lp - np.log(np.sum(np.exp(lp), axis=-1, keepdims=True))
