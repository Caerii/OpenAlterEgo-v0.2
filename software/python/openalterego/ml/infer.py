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
from typing import List, Literal, Tuple

import numpy as np
import torch

from .model import OpenAlterEgoCNN
from ..dsp.filters import preprocess_basic, preprocess_streaming


PreprocessMode = Literal["offline", "streaming", "none"]


@dataclass
class LoadedModel:
    model: OpenAlterEgoCNN
    labels: List[str]
    fs: int
    channels: int
    device: torch.device
    preprocess_mode: str = "offline"


def load_model(path: str | Path) -> LoadedModel:
    ckpt = torch.load(str(path), map_location="cpu")
    labels: List[str] = ckpt["labels"]
    fs: int = int(ckpt["fs"])
    channels: int = int(ckpt["channels"])
    preprocess_mode: str = str(ckpt.get("preprocess_mode", "offline"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = OpenAlterEgoCNN(channels=channels, classes=len(labels))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    return LoadedModel(model=model, labels=labels, fs=fs, channels=channels, device=device, preprocess_mode=preprocess_mode)


def predict_preprocessed(lm: LoadedModel, segment_preprocessed: np.ndarray) -> Tuple[str, float]:
    """Predict a single segment (already preprocessed).

    segment_preprocessed: (time, channels) float
    """
    if segment_preprocessed.ndim != 2 or segment_preprocessed.shape[1] != lm.channels:
        raise ValueError(
            f"segment must have shape (time, {lm.channels}), got {segment_preprocessed.shape}"
        )

    x = torch.from_numpy(segment_preprocessed.T[None, :, :].astype(np.float32)).to(lm.device)  # (1,ch,time)
    with torch.no_grad():
        logits = lm.model(x)
        probs = torch.softmax(logits, dim=1)[0]
        idx = int(torch.argmax(probs).item())
        return lm.labels[idx], float(probs[idx].item())


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
    if preprocess_mode == "offline":
        x = preprocess_basic(segment, fs_hz=lm.fs)
    elif preprocess_mode == "streaming":
        x = preprocess_streaming(segment, fs_hz=lm.fs, channels=lm.channels)
    elif preprocess_mode == "none":
        x = segment
    else:
        raise ValueError(f"unknown preprocess_mode: {preprocess_mode}")

    return predict_preprocessed(lm, x)
