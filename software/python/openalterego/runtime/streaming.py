"""Realtime decoding utilities.

This module is about turning a continuous stream of samples into discrete token events.
It implements a pragmatic, closed-vocabulary streaming classifier:

- maintain a rolling window (e.g. 600 ms)
- run a model every stride (e.g. 100 ms)
- stabilize/debounce predictions so you don't spam outputs
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import numpy as np

from ..core.ringbuffer import RingBuffer
from ..core.types import FrameChunk, TokenEvent
from ..ml.infer import LoadedModel, predict_preprocessed


@dataclass
class StreamDecodeConfig:
    window_ms: int = 600
    stride_ms: int = 120
    min_confidence: float = 0.70
    stable_n: int = 3
    cooldown_ms: int = 250  # don't re-emit too quickly

    blank_token: Optional[str] = None  # e.g. "<silence>"


class PredictionStabilizer:
    """Debounce a stream of (token,confidence) predictions."""

    def __init__(self, cfg: StreamDecodeConfig):
        self.cfg = cfg
        self._hist: Deque[Tuple[str, float]] = deque(maxlen=int(cfg.stable_n))
        self._last_emit_token: Optional[str] = None
        self._last_emit_t: float = 0.0

    def update(self, token: str, conf: float, *, t: float, seq: int, source: str) -> Optional[TokenEvent]:
        self._hist.append((token, float(conf)))
        if len(self._hist) < self._hist.maxlen:
            return None

        # require all last N tokens to match (strict stability)
        tokens = [x[0] for x in self._hist]
        if not all(tok == tokens[0] for tok in tokens):
            return None

        tok = tokens[0]
        if self.cfg.blank_token is not None and tok == self.cfg.blank_token:
            return None

        mean_conf = float(np.mean([x[1] for x in self._hist]))
        if mean_conf < float(self.cfg.min_confidence):
            return None

        # cooldown
        if tok == self._last_emit_token and (t - self._last_emit_t) < (self.cfg.cooldown_ms / 1000.0):
            return None

        self._last_emit_token = tok
        self._last_emit_t = float(t)
        return TokenEvent(token=tok, confidence=mean_conf, t=float(t), seq=int(seq), source=str(source), meta={})


class StreamingClassifier:
    """Run a closed-vocab classifier on a streaming signal."""

    def __init__(self, *, model: LoadedModel, cfg: StreamDecodeConfig, source_name: str = "stream"):
        self.model = model
        self.cfg = cfg
        self.source_name = str(source_name)

        self.window_samples = max(1, int(model.fs * cfg.window_ms / 1000))
        self.stride_samples = max(1, int(model.fs * cfg.stride_ms / 1000))

        # Keep 2x window so we can tolerate jittery chunk sizes.
        self.buf = RingBuffer(capacity=self.window_samples * 2, channels=model.channels, dtype=np.float32)
        self._since_last = 0
        self._stabilizer = PredictionStabilizer(cfg)

        # time/seq tracking
        self._last_seq_end: Optional[int] = None
        self._last_t_end: float = 0.0

    def push(self, chunk: FrameChunk) -> List[TokenEvent]:
        if chunk.samples.shape[1] != self.model.channels:
            raise ValueError(
                f"channel mismatch: chunk has {chunk.samples.shape[1]}, model expects {self.model.channels}"
            )

        self.buf.append(chunk.samples)
        n = int(chunk.samples.shape[0])
        self._since_last += n

        # best-effort end time/seq
        self._last_seq_end = int(chunk.seq0) + n - 1
        self._last_t_end = float(chunk.t0) + (n / float(chunk.fs_hz))

        out: List[TokenEvent] = []
        while self.buf.filled >= self.window_samples and self._since_last >= self.stride_samples:
            self._since_last -= self.stride_samples

            window = self.buf.get_last(self.window_samples)  # (time, ch) PREPROCESSED
            tok, conf = predict_preprocessed(self.model, window)

            ev = self._stabilizer.update(
                tok,
                conf,
                t=self._last_t_end,
                seq=int(self._last_seq_end or 0),
                source=self.source_name,
            )
            if ev is not None:
                out.append(ev)

        return out
