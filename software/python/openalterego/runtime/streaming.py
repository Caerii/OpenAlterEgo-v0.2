"""Realtime decoding utilities.

This module is about turning a continuous stream of samples into discrete token events.
It implements a pragmatic, closed-vocabulary streaming classifier:

- maintain a rolling window (e.g. 600 ms)
- run a model every stride (e.g. 100 ms)
- stabilize/debounce predictions so you don't spam outputs
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

from ..core.ringbuffer import RingBuffer
from ..core.types import FrameChunk, TokenEvent
from ..ml.infer import LoadedModel, predict_preprocessed_with_abstain


@dataclass
class StreamDecodeConfig:
    window_ms: int = 600
    stride_ms: int = 120
    min_confidence: float = 0.70
    stable_n: int = 3
    cooldown_ms: int = 250  # don't re-emit too quickly

    blank_token: Optional[str] = None  # e.g. "<silence>"

    # Adaptive thresholding (optional)
    adaptive_threshold: bool = False
    """If True, nudge the effective confidence gate using an EMA of recent max-probs."""
    threshold_alpha: float = 0.1
    """EMA coefficient in (0, 1] for tracking recent confidence."""
    threshold_clip_low: float = 0.5
    threshold_clip_high: float = 0.95
    baseline_snr_db: Optional[float] = None
    """If set with per-step ``snr_db``, raises the gate when SNR is below this baseline."""
    # Uncertainty abstention (optional). When triggered, stabilizer history is cleared — no token.
    abstain_entropy_norm_max: Optional[float] = None
    """If set, abstain when normalized softmax entropy exceeds this (in ``[0,1]``)."""
    abstain_min_margin: Optional[float] = None
    """If set, abstain when ``p_top1 - p_top2`` is below this."""
    snr_deficit_scale: float = 0.015
    """Added to the confidence gate per 1 dB SNR below baseline (before cap)."""
    snr_gate_cap: float = 0.22
    """Maximum extra gate from SNR deficit."""
    latency_log_every_windows: int = 0
    """If > 0, log mean ``predict_preprocessed`` time every N windows."""


class PredictionStabilizer:
    """Debounce a stream of (token,confidence) predictions."""

    def __init__(self, cfg: StreamDecodeConfig):
        self.cfg = cfg
        self._hist: Deque[Tuple[str, float]] = deque(maxlen=int(cfg.stable_n))
        self._last_emit_token: Optional[str] = None
        self._last_emit_t: float = 0.0
        self._base_threshold = float(cfg.min_confidence)
        self._eff_threshold = float(cfg.min_confidence)
        self._conf_ema: Optional[float] = None

    def reset(self) -> None:
        self._hist.clear()
        self._last_emit_token = None
        self._last_emit_t = 0.0
        self._eff_threshold = float(self.cfg.min_confidence)
        self._conf_ema = None

    def _snr_gate_delta(self, snr_db: Optional[float]) -> float:
        if self.cfg.baseline_snr_db is None or snr_db is None:
            return 0.0
        deficit = float(self.cfg.baseline_snr_db) - float(snr_db)
        if deficit <= 0.0:
            return 0.0
        return min(float(self.cfg.snr_gate_cap), deficit * float(self.cfg.snr_deficit_scale))

    def _step_adaptive_threshold(self, conf: float) -> None:
        if not self.cfg.adaptive_threshold:
            self._eff_threshold = self._base_threshold
            return
        a = min(1.0, max(1e-6, float(self.cfg.threshold_alpha)))
        c = float(conf)
        if self._conf_ema is None:
            self._conf_ema = c
        else:
            self._conf_ema = (1.0 - a) * self._conf_ema + a * c
        assert self._conf_ema is not None
        ema = self._conf_ema
        step = 0.02 * max(a, 0.05)
        lo, hi = float(self.cfg.threshold_clip_low), float(self.cfg.threshold_clip_high)
        if ema > 0.85:
            self._eff_threshold = max(lo, self._eff_threshold - step)
        elif ema < 0.60:
            self._eff_threshold = min(hi, self._eff_threshold + step)
        else:
            # drift back toward calibrated base when mid-band
            self._eff_threshold = (1.0 - 0.05 * a) * self._eff_threshold + (0.05 * a) * self._base_threshold
        self._eff_threshold = float(np.clip(self._eff_threshold, lo, hi))

    def effective_threshold(self) -> float:
        return float(self._eff_threshold)

    def update(
        self,
        token: str,
        conf: float,
        *,
        t: float,
        seq: int,
        source: str,
        snr_db: Optional[float] = None,
        abstain: bool = False,
    ) -> Optional[TokenEvent]:
        if abstain:
            self._hist.clear()
            return None
        self._step_adaptive_threshold(float(conf))
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
        gate = self._eff_threshold + self._snr_gate_delta(snr_db)
        gate = float(np.clip(gate, self.cfg.threshold_clip_low, self.cfg.threshold_clip_high + self.cfg.snr_gate_cap))
        if mean_conf < gate:
            return None

        # cooldown
        if tok == self._last_emit_token and (t - self._last_emit_t) < (self.cfg.cooldown_ms / 1000.0):
            return None

        self._last_emit_token = tok
        self._last_emit_t = float(t)
        meta: Dict[str, float] = {
            "gate_threshold": float(gate),
            "eff_threshold": float(self._eff_threshold),
        }
        if snr_db is not None:
            meta["snr_db"] = float(snr_db)
        return TokenEvent(token=tok, confidence=mean_conf, t=float(t), seq=int(seq), source=str(source), meta=meta)


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
        # Rolling chunk SNR (~0.5–1 s of acquisition at typical chunk rates) for stabilizer gate
        self._snr_hist: Deque[float] = deque(maxlen=16)

        # time/seq tracking
        self._last_seq_end: Optional[int] = None
        self._last_t_end: float = 0.0

        self._lat_n = 0
        self._lat_sum_s = 0.0

    def push(self, chunk: FrameChunk) -> List[TokenEvent]:
        if chunk.samples.shape[1] != self.model.channels:
            raise ValueError(
                f"channel mismatch: chunk has {chunk.samples.shape[1]}, model expects {self.model.channels}"
            )

        if isinstance(chunk.meta, dict):
            raw_snr = chunk.meta.get("snr_db")
            if raw_snr is not None:
                try:
                    self._snr_hist.append(float(raw_snr))
                except (TypeError, ValueError):
                    pass

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
            if int(self.cfg.latency_log_every_windows) > 0:
                t_inf0 = time.perf_counter()
            tok, conf, abstain = predict_preprocessed_with_abstain(
                self.model,
                window,
                abstain_entropy_norm_max=self.cfg.abstain_entropy_norm_max,
                abstain_min_margin=self.cfg.abstain_min_margin,
            )
            if int(self.cfg.latency_log_every_windows) > 0:
                self._lat_n += 1
                self._lat_sum_s += time.perf_counter() - t_inf0
                k = int(self.cfg.latency_log_every_windows)
                if self._lat_n % k == 0:
                    mean_ms = (self._lat_sum_s / float(k)) * 1000.0
                    log.info("inference windows: last %d mean_ms=%.3f", k, mean_ms)
                    self._lat_sum_s = 0.0

            snr_for_gate: Optional[float] = None
            if self._snr_hist:
                snr_for_gate = float(np.mean(np.asarray(list(self._snr_hist), dtype=np.float64)))

            ev = self._stabilizer.update(
                tok,
                conf,
                t=self._last_t_end,
                seq=int(self._last_seq_end or 0),
                source=self.source_name,
                snr_db=snr_for_gate,
                abstain=abstain,
            )
            if ev is not None:
                out.append(ev)

        return out
