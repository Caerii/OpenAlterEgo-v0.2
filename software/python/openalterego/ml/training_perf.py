"""Training throughput helpers (DataLoader, CUDA knobs, optional compile)."""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def resolve_use_amp(device: torch.device, *, no_amp: bool = False) -> bool:
    """AMP on by default when training on CUDA."""
    return device.type == "cuda" and not bool(no_amp)


def cuda_synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@dataclass
class EpochTiming:
    """Per-epoch wall times (seconds)."""

    data_wait_s: float = 0.0
    compute_s: float = 0.0
    val_s: float = 0.0

    @property
    def train_s(self) -> float:
        return self.data_wait_s + self.compute_s

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainPhaseTiming:
    """Setup + training phase breakdown for benchmarks."""

    load_signals_s: float = 0.0
    preprocess_s: float = 0.0
    preprocess_cache_hit: bool = False
    segment_train_s: float = 0.0
    segment_train_cache_hit: bool = False
    segment_val_s: float = 0.0
    segment_val_cache_hit: bool = False
    dataloader_init_s: float = 0.0
    model_init_s: float = 0.0
    epoch_timings: List[EpochTiming] = field(default_factory=list)
    device: str = "cpu"
    use_amp: bool = False
    num_workers: int = 0
    n_train: int = 0
    n_val: int = 0
    batch_size: int = 32

    def bottleneck_label(self) -> str:
        """Heuristic dominant phase from the last epoch (or setup if no epochs)."""
        if self.epoch_timings:
            ep = self.epoch_timings[-1]
            parts = {
                "dataloader": ep.data_wait_s,
                "compute": ep.compute_s,
                "validation": ep.val_s,
            }
            return max(parts, key=parts.get)  # type: ignore[arg-type]
        setup = {
            "preprocess": self.preprocess_s,
            "segment_train": self.segment_train_s,
            "segment_val": self.segment_val_s,
            "load_signals": self.load_signals_s,
        }
        return max(setup, key=setup.get)  # type: ignore[arg-type]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bottleneck"] = self.bottleneck_label()
        if self.epoch_timings:
            ep = self.epoch_timings[-1]
            d["last_epoch_train_s"] = ep.train_s
            d["last_epoch_val_s"] = ep.val_s
            total = ep.train_s + ep.val_s
            if total > 0:
                d["last_epoch_data_pct"] = round(100.0 * ep.data_wait_s / total, 1)
                d["last_epoch_compute_pct"] = round(100.0 * ep.compute_s / total, 1)
                d["last_epoch_val_pct"] = round(100.0 * ep.val_s / total, 1)
        return d


def default_num_workers(requested: int) -> int:
    """Pick safe default worker count when ``requested < 0`` (auto)."""
    if int(requested) >= 0:
        return int(requested)
    # Windows uses spawn; worker IPC often loses vs main-thread for small EMG sessions.
    if os.name == "nt":
        return 0
    cpu = os.cpu_count() or 4
    return max(0, min(8, cpu - 1))


def build_train_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    device: torch.device,
    num_workers: int = -1,
    shuffle: bool = True,
) -> DataLoader:
    nw = default_num_workers(int(num_workers))
    pin = device.type == "cuda"
    kwargs: Dict[str, Any] = {
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "drop_last": False,
        "pin_memory": pin,
        "num_workers": nw,
    }
    if nw > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def maybe_compile_model(model: nn.Module, *, enabled: bool) -> nn.Module:
    if not enabled or device_type_cuda_unavailable():
        return model
    if not hasattr(torch, "compile"):
        return model
    try:
        return torch.compile(model)  # type: ignore[return-value]
    except Exception:
        return model


def device_type_cuda_unavailable() -> bool:
    return not torch.cuda.is_available()


def configure_tf32(enabled: bool = True) -> None:
    """Enable TF32 matmul on Ampere+ GPUs (faster, usually negligible accuracy impact)."""
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = bool(enabled)
    torch.backends.cudnn.allow_tf32 = bool(enabled)
