"""Push-to-talk utterance buffering for open-speech CTC decode."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional

import numpy as np


@dataclass
class UtteranceSegmenterConfig:
    fs_hz: int
    channels: int
    pad_ms: int = 150
    min_utterance_ms: int = 200
    max_utterance_ms: int = 15000


@dataclass
class UtteranceSegmenter:
    """Buffer EMG samples between PTT start/end."""

    cfg: UtteranceSegmenterConfig
    state: Literal["idle", "recording"] = "idle"
    _chunks: List[np.ndarray] = field(default_factory=list)
    _pre_pad: List[np.ndarray] = field(default_factory=list)
    _pre_pad_max: int = field(default=8)

    def on_ptt_start(self) -> None:
        self.state = "recording"
        self._chunks = [c.copy() for c in self._pre_pad[-self._pre_pad_max :]]

    def on_ptt_end(self) -> Optional[np.ndarray]:
        self.state = "idle"
        if not self._chunks:
            return None
        out = np.concatenate(self._chunks, axis=0).astype(np.float32, copy=False)
        self._chunks = []
        min_n = int(self.cfg.min_utterance_ms * self.cfg.fs_hz / 1000)
        max_n = int(self.cfg.max_utterance_ms * self.cfg.fs_hz / 1000)
        if out.shape[0] < min_n:
            return None
        if out.shape[0] > max_n:
            out = out[:max_n, :]
        return out

    def feed(self, chunk: np.ndarray) -> None:
        x = np.asarray(chunk, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError("chunk must be (T, C)")
        if self.state == "recording":
            self._chunks.append(x.copy())
        self._pre_pad.append(x.copy())
        if len(self._pre_pad) > self._pre_pad_max * 3:
            self._pre_pad = self._pre_pad[-self._pre_pad_max :]
