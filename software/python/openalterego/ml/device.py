"""Training/inference device selection."""

from __future__ import annotations

import torch


def resolve_device(preferred: str = "auto") -> torch.device:
    """Pick compute device. ``auto`` uses CUDA when available."""
    key = str(preferred or "auto").strip().lower()
    if key == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if key == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but not available. Install a CUDA PyTorch build "
                "(see software/python/pyproject.toml) and verify drivers."
            )
        return torch.device("cuda")
    if key == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unknown device: {preferred!r} (use auto, cuda, or cpu)")


def configure_cuda_for_training(device: torch.device, *, tf32: bool = True) -> None:
    """Enable common CUDA throughput knobs (no-op on CPU)."""
    if device.type != "cuda":
        return
    torch.backends.cudnn.benchmark = True
    if tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
