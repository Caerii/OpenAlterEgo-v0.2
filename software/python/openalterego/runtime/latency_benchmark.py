"""End-to-end latency benchmarking for the realtime pipeline (p50/p95/p99)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ..acquisition.simulate import SimConfig, stream_simulated_chunks
from ..core.types import FrameChunk
from ..dsp.emg_config import build_online_preprocessor, resolve_emg_mode_for_serve
from ..ml.infer import LoadedModel, load_model, predict_preprocessed_with_abstain
from ..runtime.streaming import StreamDecodeConfig, StreamingClassifier


@dataclass(frozen=True)
class LatencyPercentiles:
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    n: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "p50_ms": round(self.p50_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "p99_ms": round(self.p99_ms, 3),
            "mean_ms": round(self.mean_ms, 3),
            "n": int(self.n),
        }


@dataclass
class LatencyReport:
    """Timing breakdown for preprocess, inference window, and chunk push."""

    fs_hz: int
    channels: int
    window_ms: int
    stride_ms: int
    n_chunks: int
    preprocess: LatencyPercentiles
    inference_window: LatencyPercentiles
    chunk_push: LatencyPercentiles
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fs_hz": self.fs_hz,
            "channels": self.channels,
            "window_ms": self.window_ms,
            "stride_ms": self.stride_ms,
            "n_chunks": self.n_chunks,
            "preprocess": self.preprocess.to_dict(),
            "inference_window": self.inference_window.to_dict(),
            "chunk_push": self.chunk_push.to_dict(),
            "notes": list(self.notes),
        }


def _percentiles_ms(samples_s: List[float]) -> LatencyPercentiles:
    if not samples_s:
        return LatencyPercentiles(0.0, 0.0, 0.0, 0.0, 0)
    arr = np.asarray(samples_s, dtype=np.float64) * 1000.0
    return LatencyPercentiles(
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        mean_ms=float(np.mean(arr)),
        n=int(arr.size),
    )


def run_latency_benchmark(
    *,
    model_path: str,
    fs_hz: Optional[int] = None,
    channels: Optional[int] = None,
    window_ms: int = 600,
    stride_ms: int = 120,
    n_chunks: int = 200,
    seed: int = 0,
    sim_engine: str = "heuristic",
    warmup_chunks: int = 8,
    motion_gate: bool = False,
    motion_threshold: float = 0.35,
) -> LatencyReport:
    """Measure preprocess / inference / push latencies on simulated chunks."""
    lm = load_model(str(model_path))
    fs = int(fs_hz if fs_hz is not None else lm.fs)
    ch = int(channels if channels is not None else lm.channels)
    emg_mode = resolve_emg_mode_for_serve(checkpoint_emg_mode=lm.emg_mode, profile_preprocessing_mode=None)
    pre = build_online_preprocessor(
        fs_hz=float(fs),
        channels=ch,
        emg_mode=emg_mode,
        motion_gate=bool(motion_gate),
        motion_threshold=float(motion_threshold),
    )
    decode_cfg = StreamDecodeConfig(window_ms=int(window_ms), stride_ms=int(stride_ms), stable_n=1, min_confidence=0.0)
    clf = StreamingClassifier(model=lm, cfg=decode_cfg, source_name="latency_bench")

    sim = SimConfig(
        fs_hz=fs,
        channels=ch,
        seed=int(seed),
        realtime_clock=False,
        sim_engine=str(sim_engine),
    )

    pre_times: List[float] = []
    inf_times: List[float] = []
    push_times: List[float] = []
    i = 0

    for i, chunk in enumerate(stream_simulated_chunks(sim)):
        if i >= int(n_chunks) + int(warmup_chunks):
            break
        t0 = time.perf_counter()
        y = pre.process(chunk.samples)
        t_pre = time.perf_counter() - t0

        meta = dict(chunk.meta)
        if pre.last_motion_gated:
            meta["motion_gated"] = True
        fc = FrameChunk(samples=y, fs_hz=chunk.fs_hz, t0=chunk.t0, seq0=chunk.seq0, meta=meta)

        t1 = time.perf_counter()
        _ = clf.push(fc)
        t_push = time.perf_counter() - t1

        if i >= int(warmup_chunks):
            pre_times.append(t_pre)
            push_times.append(t_push)

        if clf.buf.filled >= clf.window_samples:
            window = clf.buf.get_last(clf.window_samples)
            t2 = time.perf_counter()
            predict_preprocessed_with_abstain(lm, window)
            inf_times.append(time.perf_counter() - t2)

    report = LatencyReport(
        fs_hz=fs,
        channels=ch,
        window_ms=int(window_ms),
        stride_ms=int(stride_ms),
        n_chunks=int(min(n_chunks, max(0, i + 1 - warmup_chunks))),
        preprocess=_percentiles_ms(pre_times),
        inference_window=_percentiles_ms(inf_times[-max(1, n_chunks * 2) :]),
        chunk_push=_percentiles_ms(push_times),
    )
    e2e_p95 = report.preprocess.p95_ms + report.inference_window.p95_ms
    report.notes.append(f"approx_e2e_p95_ms={e2e_p95:.1f} (preprocess + inference window)")
    if e2e_p95 > 500.0:
        report.notes.append("WARN: p95 exceeds 500 ms literature HCI target")
    else:
        report.notes.append("OK: p95 within 500 ms literature target")
    return report
