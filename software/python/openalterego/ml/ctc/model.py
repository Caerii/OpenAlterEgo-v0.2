"""GRU + CTC model for per-word phoneme sequences."""

from __future__ import annotations

import torch
import torch.nn as nn


class GowdaCTCModel(nn.Module):
    """CNN downsample → BiGRU → per-frame phoneme logits (incl. blank)."""

    def __init__(self, channels: int, n_phonemes: int, *, n_frames: int = 40, hidden: int = 256):
        super().__init__()
        self.n_frames = int(n_frames)
        self.stem = nn.Sequential(
            nn.Conv1d(int(channels), 64, kernel_size=7, stride=4, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=5, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(self.n_frames)
        self.gru = nn.GRU(128, int(hidden), num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(int(hidden) * 2, int(n_phonemes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        h = self.stem(x)
        h = self.pool(h)
        h = h.transpose(1, 2)  # (B, T', C')
        h, _ = self.gru(h)
        return self.fc(h)  # (B, T', n_phonemes)


class GowdaSPDCTCModelLegacy(nn.Module):
    """Phase-3 SPD GRU (2 layers, no LayerNorm) for checkpoint compatibility."""

    def __init__(self, feature_dim: int, n_phonemes: int, *, hidden: int = 256, num_layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(
            int(feature_dim),
            int(hidden),
            num_layers=int(num_layers),
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(int(hidden) * 2, int(n_phonemes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(x)
        return self.fc(h)


class GowdaSPDCTCModel(nn.Module):
    """LayerNorm + deep BiGRU on σ(τ) sequences (paper: 3-layer GRU; +dropout beyond paper)."""

    def __init__(
        self,
        feature_dim: int,
        n_phonemes: int,
        *,
        hidden: int = 256,
        num_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(int(feature_dim))
        self.gru = nn.GRU(
            int(feature_dim),
            int(hidden),
            num_layers=int(num_layers),
            batch_first=True,
            bidirectional=True,
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
        )
        self.out_norm = nn.LayerNorm(int(hidden) * 2)
        self.dropout = nn.Dropout(float(dropout))
        self.fc = nn.Linear(int(hidden) * 2, int(n_phonemes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        h = self.input_norm(x)
        h, _ = self.gru(h)
        h = self.out_norm(h)
        h = self.dropout(h)
        return self.fc(h)
