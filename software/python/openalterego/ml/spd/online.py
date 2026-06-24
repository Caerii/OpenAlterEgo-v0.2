"""Causal online SPD σ(τ) frame stream for utterance decode."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ...dsp.filters import preprocess_basic
from .features import (
    GOWDA_SPD_ETA,
    GOWDA_SPD_STEP_MS,
    GOWDA_SPD_WINDOW_MS,
    append_sigma_deltas,
    edge_matrix,
    sigma_vector,
    sliding_edge_matrices,
    spd_regularize,
    step_samples,
    window_samples,
)


class OnlineSPDStream:
    """Incremental σ(τ) generator; ``build_sequence`` matches offline ``segment_to_sigma_sequence``."""

    def __init__(
        self,
        basis_q: np.ndarray,
        *,
        fs_hz: int,
        channels: int,
        window_ms: int = GOWDA_SPD_WINDOW_MS,
        step_ms: int = GOWDA_SPD_STEP_MS,
        eta: float = GOWDA_SPD_ETA,
        feature_mode: str = "diag_delta",
        emg_mode: str = "gowda",
    ):
        self._q = np.asarray(basis_q, dtype=np.float64)
        self._fs = int(fs_hz)
        self._ch = int(channels)
        self._win = window_samples(self._fs, int(window_ms))
        self._step = step_samples(self._fs, int(step_ms))
        self._eta = float(eta)
        self._feature_mode = str(feature_mode)
        self._emg_mode = str(emg_mode)
        self._buf = np.zeros((0, self._ch), dtype=np.float32)
        self._next_emit_start = 0
        self._frames: List[np.ndarray] = []

    def reset(self) -> None:
        self._buf = np.zeros((0, self._ch), dtype=np.float32)
        self._next_emit_start = 0
        self._frames = []

    def _preprocess(self, x: np.ndarray) -> np.ndarray:
        return preprocess_basic(
            np.asarray(x, dtype=np.float32),
            fs_hz=self._fs,
            mode=self._emg_mode,  # type: ignore[arg-type]
            rectify_signals=False,
            normalize_mode="zscore",
        )

    def _emit_from_buffer(self) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        n = int(self._buf.shape[0])
        while self._next_emit_start + self._win <= n:
            s0 = int(self._next_emit_start)
            s1 = s0 + self._win
            e = spd_regularize(edge_matrix(self._buf[s0:s1, :]), eta=self._eta)
            out.append(
                sigma_vector(e, self._q, feature_mode=self._feature_mode).astype(np.float32, copy=False)
            )
            self._frames.append(out[-1])
            self._next_emit_start += self._step
        return out

    def push(self, samples: np.ndarray) -> List[np.ndarray]:
        """Append ``(T, C)`` samples; return new σ frames since last push."""
        x = self._preprocess(samples)
        if x.size == 0:
            return []
        if self._buf.size == 0:
            self._buf = x
        else:
            self._buf = np.concatenate([self._buf, x], axis=0)
        return self._emit_from_buffer()

    def flush(self) -> List[np.ndarray]:
        """Pad tail and emit remaining frames (utterance end)."""
        if self._buf.shape[0] < self._win:
            pad = self._win - int(self._buf.shape[0])
            if pad > 0:
                self._buf = np.concatenate(
                    [self._buf, np.zeros((pad, self._ch), dtype=np.float32)], axis=0
                )
        return self._emit_from_buffer()

    def build_sequence(self, utterance_time_ch: np.ndarray) -> np.ndarray:
        """Full utterance σ sequence — must match offline SPD pipeline."""
        from .features import segment_to_sigma_sequence

        seg = self._preprocess(utterance_time_ch)
        seq = segment_to_sigma_sequence(
            seg,
            self._q,
            fs_hz=self._fs,
            eta=self._eta,
            feature_mode=self._feature_mode,
        )
        return np.asarray(seq, dtype=np.float32)

    def frames_as_sequence(self) -> np.ndarray:
        """Stack frames accumulated via push/flush."""
        if not self._frames:
            d = int(sigma_vector(np.eye(self._ch), self._q, feature_mode=self._feature_mode).size)
            if self._feature_mode == "diag_delta":
                d *= 2
            return np.zeros((1, d), dtype=np.float32)
        seq = np.stack(self._frames, axis=0).astype(np.float32, copy=False)
        if self._feature_mode == "diag_delta":
            seq = append_sigma_deltas(seq)
        return seq
