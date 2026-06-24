"""Shared CTC batch helpers."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch


def unpack_batch(
    batch: Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ],
):
    if len(batch) == 4:
        x, targets, t_lens, x_lens = batch
        return x, targets, t_lens, x_lens
    x, targets, t_lens = batch
    return x, targets, t_lens, None


def input_lengths(logits: torch.Tensor, x_lens: Optional[torch.Tensor]) -> torch.Tensor:
    if x_lens is not None:
        return x_lens.to(device=logits.device, dtype=torch.long)
    return torch.full((logits.size(0),), logits.size(1), dtype=torch.long, device=logits.device)
