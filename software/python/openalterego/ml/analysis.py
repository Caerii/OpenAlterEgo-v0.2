"""Model analysis utilities (channel importance, etc.)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .infer import load_model
from .model import SEBlock1d, create_model
from .train import SegmentDataset


@dataclass
class ChannelImportanceReport:
    session_dir: str
    model_path: str
    fs_hz: int
    channels: int
    n_samples: int
    se_scores: List[float]
    grad_scores: List[float]
    combined_scores: List[float]
    top_channels: List[int] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_dir": self.session_dir,
            "model_path": self.model_path,
            "fs_hz": int(self.fs_hz),
            "channels": int(self.channels),
            "n_samples": int(self.n_samples),
            "se_scores": [round(float(x), 6) for x in self.se_scores],
            "grad_scores": [round(float(x), 6) for x in self.grad_scores],
            "combined_scores": [round(float(x), 6) for x in self.combined_scores],
            "top_channels": [int(x) for x in self.top_channels],
            "notes": list(self.notes),
        }


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64)
    if s.size == 0:
        return s
    mx = float(np.max(s))
    if mx <= 1e-12:
        return np.zeros_like(s)
    return s / mx


@torch.no_grad()
def _se_channel_scores(model: nn.Module, x: torch.Tensor) -> np.ndarray:
    """Input-channel proxy from SE gates at the stem (same width as EMG channels)."""
    n_in = int(x.shape[1])
    gates: List[torch.Tensor] = []

    def _hook(module: nn.Module, inp: tuple[torch.Tensor, ...], _out: torch.Tensor) -> None:
        x_in = inp[0]
        if int(x_in.shape[1]) != n_in:
            return
        b, c, _ = x_in.shape
        w = module.pool(x_in).view(b, c)
        w = module.fc(w)
        gates.append(w.detach().cpu())

    hooks = []
    for mod in model.modules():
        if isinstance(mod, SEBlock1d):
            hooks.append(mod.register_forward_hook(_hook))
    try:
        model(x)
    finally:
        for h in hooks:
            h.remove()
    if not gates:
        return np.zeros(n_in, dtype=np.float64)
    stacked = torch.stack(gates, dim=0).mean(dim=0).mean(dim=0).numpy()
    return stacked.astype(np.float64, copy=False)


def _grad_channel_scores(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> np.ndarray:
    model.eval()
    x = x.detach().clone().requires_grad_(True)
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    grad = x.grad
    if grad is None:
        return np.zeros(int(x.shape[1]), dtype=np.float64)
    return grad.abs().mean(dim=(0, 2)).detach().cpu().numpy().astype(np.float64, copy=False)


def compute_channel_importance(
    model: nn.Module,
    dl: DataLoader,
    device: torch.device,
    *,
    max_batches: int = 32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (se_scores, grad_scores, combined_scores) length ``channels``."""
    se_acc = None
    grad_acc = None
    n = 0
    for bi, (x, y) in enumerate(dl):
        if bi >= int(max_batches):
            break
        x = x.to(device)
        y = y.to(device)
        se = _se_channel_scores(model, x)
        grad = _grad_channel_scores(model, x, y)
        if se_acc is None:
            se_acc = np.zeros_like(se)
            grad_acc = np.zeros_like(grad)
        se_acc += se
        grad_acc += grad
        n += 1
    if n == 0 or se_acc is None or grad_acc is None:
        ch = 0
        z = np.zeros(0, dtype=np.float64)
        return z, z, z
    se_acc /= float(n)
    grad_acc /= float(n)
    if int(se_acc.shape[0]) != int(grad_acc.shape[0]):
        combined = _normalize_scores(grad_acc)
    else:
        combined = 0.5 * _normalize_scores(se_acc) + 0.5 * _normalize_scores(grad_acc)
    return se_acc, grad_acc, combined


def run_channel_importance(
    session_dir: Path,
    model_path: Path,
    *,
    segment_ms: Optional[int] = None,
    batch_size: int = 16,
    max_batches: int = 32,
    top_k: int = 16,
    seed: int = 1337,
) -> ChannelImportanceReport:
    """Compute channel importance on a session using a trained checkpoint."""
    import json

    session_dir = Path(session_dir)
    model_path = Path(model_path)
    lm = load_model(model_path)
    signals = np.load(session_dir / "signals.npy")
    events = pd.read_csv(session_dir / "events.csv")
    meta: dict = {}
    meta_path = session_dir / "meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fs_hz = int(meta.get("fs_hz", lm.fs))
    ckpt = torch.load(str(model_path), map_location="cpu")
    seg_ms = int(segment_ms if segment_ms is not None else ckpt.get("segment_ms", 600))
    emg_mode = str(ckpt.get("emg_mode", lm.emg_mode or "standard"))
    prep_mode = str(ckpt.get("preprocess_mode", "streaming"))
    labels = list(ckpt["labels"])
    label_to_id = {lab: i for i, lab in enumerate(labels)}
    events = events[events["label"].astype(str).isin(label_to_id)].reset_index(drop=True)

    from ..dsp.preprocess_cache import ensure_preprocessed_signals

    prepared = signals
    if prep_mode != "none":
        prepared, _ = ensure_preprocessed_signals(
            signals,
            session_dir,
            preprocess_mode=prep_mode,  # type: ignore[arg-type]
            emg_mode=emg_mode,  # type: ignore[arg-type]
            fs_hz=int(fs_hz),
            use_cache=True,
            rebuild=False,
            show_progress=False,
        )

    ds = SegmentDataset(
        prepared,
        events,
        label_to_id,
        fs_hz=fs_hz,
        segment_ms=seg_ms,
        preprocess_mode="none",
        emg_mode=emg_mode,  # type: ignore[arg-type]
        seed=int(seed),
    )
    dl = DataLoader(ds, batch_size=int(batch_size), shuffle=True)
    se, grad, combined = compute_channel_importance(
        lm.model, dl, lm.device, max_batches=int(max_batches)
    )
    channels = int(signals.shape[1])
    if combined.size == 0:
        combined = np.zeros(channels, dtype=np.float64)
        se = combined.copy()
        grad = combined.copy()
    order = np.argsort(-combined)
    top = order[: min(int(top_k), channels)].astype(int).tolist()
    notes: List[str] = []
    if not any(isinstance(m, SEBlock1d) for m in lm.model.modules()):
        notes.append("model has no SE blocks; se_scores may be zero")
    return ChannelImportanceReport(
        session_dir=str(session_dir),
        model_path=str(model_path),
        fs_hz=int(fs_hz),
        channels=int(channels),
        n_samples=len(ds),
        se_scores=se.tolist(),
        grad_scores=grad.tolist(),
        combined_scores=combined.tolist(),
        top_channels=top,
        notes=notes,
    )


def load_teacher_models(paths: Sequence[str | Path], device: torch.device) -> List[nn.Module]:
    teachers: List[nn.Module] = []
    for p in paths:
        lm = load_model(p)
        lm.model.to(device)
        lm.model.eval()
        for param in lm.model.parameters():
            param.requires_grad = False
        teachers.append(lm.model)
    return teachers
