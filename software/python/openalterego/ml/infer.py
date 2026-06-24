"""Inference helpers.

The starter repo keeps inference intentionally simple:
- load a small CNN checkpoint
- predict on a fixed-length window

For realtime usage, you usually want:
- *streaming-compatible* preprocessing (causal filters, running normalization)
- debouncing/stabilization (see :mod:`openalterego.runtime.streaming`)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import numpy as np
import torch

from .model import create_model
from ..dsp.emg_config import EmgMode
from ..dsp.filters import preprocess_basic, preprocess_streaming


PreprocessMode = Literal["offline", "streaming", "none"]


@dataclass
class LoadedModel:
    model: torch.nn.Module
    labels: List[str]
    fs: int
    channels: int
    device: torch.device
    preprocess_mode: str = "offline"
    user_id: Optional[str] = None
    # standard/clinical/wide as trained; None = legacy checkpoint without emg_mode key
    emg_mode: Optional[EmgMode] = None
    arch: str = "cnn"


def load_model(path: str | Path) -> LoadedModel:
    ckpt = torch.load(str(path), map_location="cpu")
    labels: List[str] = ckpt["labels"]
    fs: int = int(ckpt["fs"])
    channels: int = int(ckpt["channels"])
    preprocess_mode: str = str(ckpt.get("preprocess_mode", "offline"))
    raw_emg = ckpt.get("emg_mode")
    emg_mode: Optional[EmgMode]
    if raw_emg is None or str(raw_emg).strip() == "":
        emg_mode = None
    else:
        emg_mode = str(raw_emg).strip()  # type: ignore[assignment]
    uid = ckpt.get("user_id")
    user_id: Optional[str] = str(uid) if uid is not None else None
    arch = str(ckpt.get("arch", "cnn"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model(arch, channels=channels, classes=len(labels))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    return LoadedModel(
        model=model,
        labels=labels,
        fs=fs,
        channels=channels,
        device=device,
        preprocess_mode=preprocess_mode,
        user_id=user_id,
        emg_mode=emg_mode,
        arch=arch,
    )


def _softmax_entropy_norm(probs: np.ndarray) -> float:
    """Normalized entropy in ``[0, 1]`` (max-entropy = uniform)."""
    p = np.clip(probs.astype(np.float64), 1e-12, 1.0)
    k = int(p.size)
    if k <= 1:
        return 0.0
    h = float(-np.sum(p * np.log(p)))
    return float(h / np.log(k))


def _top2_margin(probs: np.ndarray) -> float:
    if probs.size < 2:
        return 1.0
    s = np.sort(probs.astype(np.float64))[::-1]
    return float(s[0] - s[1])


def _forward_probs(lm: LoadedModel, segment_preprocessed: np.ndarray) -> Tuple[np.ndarray, int, float]:
    if segment_preprocessed.ndim != 2 or segment_preprocessed.shape[1] != lm.channels:
        raise ValueError(
            f"segment must have shape (time, {lm.channels}), got {segment_preprocessed.shape}"
        )
    x = torch.from_numpy(segment_preprocessed.T[None, :, :].astype(np.float32)).to(lm.device)
    with torch.no_grad():
        logits = lm.model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    idx = int(np.argmax(probs))
    return probs, idx, float(probs[idx])


def predict_preprocessed(lm: LoadedModel, segment_preprocessed: np.ndarray) -> Tuple[str, float]:
    """Predict a single segment (already preprocessed).

    segment_preprocessed: (time, channels) float
    """
    _, idx, conf = _forward_probs(lm, segment_preprocessed)
    return lm.labels[idx], conf


def predict_preprocessed_with_abstain(
    lm: LoadedModel,
    segment_preprocessed: np.ndarray,
    *,
    abstain_entropy_norm_max: Optional[float] = None,
    abstain_min_margin: Optional[float] = None,
) -> Tuple[str, float, bool]:
    """Like :func:`predict_preprocessed` but sets ``abstain=True`` if confidence shape looks ambiguous.

    When ``abstain`` is True, ``token`` / ``confidence`` are still the argmax (for diagnostics);
    the streaming stabilizer should treat abstain as "no update" (clear history).
    """
    probs, idx, conf = _forward_probs(lm, segment_preprocessed)
    abstain = False
    if abstain_entropy_norm_max is not None:
        if _softmax_entropy_norm(probs) > float(abstain_entropy_norm_max):
            abstain = True
    if not abstain and abstain_min_margin is not None:
        if _top2_margin(probs) < float(abstain_min_margin):
            abstain = True
    return lm.labels[idx], conf, abstain


def predict_segment(
    lm: LoadedModel,
    segment: np.ndarray,
    *,
    preprocess_mode: PreprocessMode = "offline",
) -> Tuple[str, float]:
    """Predict a single segment.

    segment: (time, channels) float

    preprocess_mode:
        - "offline": zero-phase filters (analysis / training convenience)
        - "streaming": chunked causal filters (closer to realtime)
        - "none": segment is assumed already preprocessed
    """
    emg: Optional[EmgMode] = lm.emg_mode if lm.emg_mode in ("standard", "clinical", "wide", "gowda") else None
    if preprocess_mode == "offline":
        x = preprocess_basic(segment, fs_hz=float(lm.fs), mode=emg)
    elif preprocess_mode == "streaming":
        x = preprocess_streaming(
            segment,
            fs_hz=float(lm.fs),
            channels=lm.channels,
            mode=emg,
        )
    elif preprocess_mode == "none":
        x = segment
    else:
        raise ValueError(f"unknown preprocess_mode: {preprocess_mode}")

    return predict_preprocessed(lm, x)
