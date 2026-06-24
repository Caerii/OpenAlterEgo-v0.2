"""ML models for OpenAlterEgo.

Baseline: 1D CNN over multichannel time series.
Upgrade: 1D SE-ResNet with channel squeeze-excitation (Tang et al. 2025).

Input shape: (batch, channels, time)
"""

from __future__ import annotations

from typing import Literal, Union

import torch
import torch.nn as nn

ModelArch = Literal["cnn", "se_resnet"]


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
        self.arch = "cnn"
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


class SEBlock1d(nn.Module):
    """Squeeze-and-excitation over temporal mean (channel recalibration)."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(4, channels // int(reduction))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1)
        return x * w


class SEResBlock1d(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        kernel: int = 5,
        stride: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=kernel, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.se = SEBlock1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=dropout)
        self.downsample: nn.Module
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.drop(out)
        out = out + identity
        return self.relu(out)


class OpenAlterEgoSEResNet(nn.Module):
    """1D SE-ResNet baseline aligned with Tang 2025 (channel-attentive ResNet)."""

    def __init__(self, channels: int, classes: int):
        super().__init__()
        self.arch = "se_resnet"
        self.stem = nn.Sequential(
            nn.Conv1d(channels, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.layer1 = nn.Sequential(
            SEResBlock1d(64, 64, kernel=7),
            SEResBlock1d(64, 64, kernel=5),
        )
        self.layer2 = nn.Sequential(
            SEResBlock1d(64, 128, kernel=5, stride=2),
            SEResBlock1d(128, 128, kernel=5),
        )
        self.layer3 = nn.Sequential(
            SEResBlock1d(128, 256, kernel=3, stride=2, dropout=0.15),
            SEResBlock1d(256, 256, kernel=3, dropout=0.15),
        )
        self.layer4 = nn.Sequential(
            SEResBlock1d(256, 256, kernel=3, stride=2, dropout=0.2),
            SEResBlock1d(256, 256, kernel=3, dropout=0.2),
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
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.head(x)


def create_model(arch: str, channels: int, classes: int) -> nn.Module:
    """Factory for supported model architectures."""
    key = str(arch or "cnn").strip().lower()
    if key in ("cnn", "baseline"):
        return OpenAlterEgoCNN(channels=int(channels), classes=int(classes))
    if key in ("se_resnet", "seresnet", "se-resnet"):
        return OpenAlterEgoSEResNet(channels=int(channels), classes=int(classes))
    raise ValueError(f"unknown model arch: {arch!r} (use cnn or se_resnet)")


def default_arch() -> ModelArch:
    return "se_resnet"
