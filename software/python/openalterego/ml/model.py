"""ML models for OpenAlterEgo (baseline).

Baseline: 1D CNN over multichannel time series.
Input shape: (batch, channels, time)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 5, pool: int = 2, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OpenAlterEgoCNN(nn.Module):
    def __init__(self, channels: int, classes: int):
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBlock(channels, 64, k=7, pool=2, dropout=0.1),
            ConvBlock(64, 128, k=5, pool=2, dropout=0.1),
            ConvBlock(128, 256, k=3, pool=2, dropout=0.2),
            ConvBlock(256, 256, k=3, pool=2, dropout=0.2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.head(x)
        return x
