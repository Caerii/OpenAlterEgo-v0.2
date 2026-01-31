"""Transport simulation: jitter + packet loss.

This is useful for stress-testing your decoder and timing logic.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import AsyncIterator, Iterable, Iterator, Optional

import numpy as np


@dataclass
class LinkConfig:
    loss_prob: float = 0.0
    jitter_ms: float = 0.0
    extra_latency_ms: float = 0.0
    seed: int = 123

    def __post_init__(self) -> None:
        if self.loss_prob < 0 or self.loss_prob > 1:
            raise ValueError("loss_prob must be in [0,1]")
        if self.jitter_ms < 0:
            raise ValueError("jitter_ms must be >=0")
        if self.extra_latency_ms < 0:
            raise ValueError("extra_latency_ms must be >=0")


async def apply_link(
    packets: AsyncIterator[bytes],
    *,
    cfg: LinkConfig,
) -> AsyncIterator[bytes]:
    """Async generator that drops / delays packets."""
    rng = np.random.default_rng(int(cfg.seed))

    async for p in packets:
        if cfg.loss_prob > 0 and rng.random() < cfg.loss_prob:
            continue

        delay = float(cfg.extra_latency_ms) / 1000.0
        if cfg.jitter_ms > 0:
            delay += float(rng.normal(scale=cfg.jitter_ms / 1000.0))
            delay = max(0.0, delay)
        if delay > 0:
            await asyncio.sleep(delay)
        yield p
