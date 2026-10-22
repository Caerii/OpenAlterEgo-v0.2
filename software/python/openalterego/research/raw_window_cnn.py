"""Small temporal CNN control for fixed-channel SilentWear event windows."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class RawWindowCNN(nn.Module):
    def __init__(self, channels: int = 6, classes: int = 9) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(channels, 24, 9, stride=2, padding=4),
            nn.BatchNorm1d(24), nn.GELU(),
            nn.Conv1d(24, 24, 7, stride=2, padding=3, groups=24),
            nn.Conv1d(24, 48, 1), nn.BatchNorm1d(48), nn.GELU(),
            nn.Conv1d(48, 48, 7, stride=2, padding=3, groups=48),
            nn.Conv1d(48, 64, 1), nn.BatchNorm1d(64), nn.GELU(),
        )
        self.head = nn.Sequential(nn.Linear(128, 64), nn.GELU(),
                                  nn.Dropout(0.2), nn.Linear(64, classes))

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        features = self.features(signal)
        pooled = torch.cat((features.mean(dim=-1),
                            features.amax(dim=-1)), dim=1)
        return self.head(pooled)


def normalize_window_batch(windows: np.ndarray,
                           channels: tuple[int, ...]) -> np.ndarray:
    """Unlabeled offline per-recording amplitude normalization."""
    if (windows.ndim != 3 or windows.shape[1:] != (400, 14) or
            len(channels) != 6 or len(set(channels)) != 6 or
            not np.isfinite(windows).all()):
        raise ValueError("expected finite 400-sample, 14-channel event windows")
    selected = windows[:, :, channels].astype(np.float32)
    scale = np.maximum(np.median(np.abs(selected), axis=(0, 1)), 1e-6)
    return np.clip(selected / scale[None, None, :], -8, 8).transpose(0, 2, 1).copy()
