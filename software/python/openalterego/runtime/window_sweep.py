"""Window size sweep: latency vs event-level accuracy trade-offs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..dsp.emg_config import resolve_emg_mode_for_serve
from ..ml.infer import load_model, predict_preprocessed
from ..dsp.filters import preprocess_streaming
from .latency_benchmark import run_latency_benchmark


@dataclass(frozen=True)
class WindowSweepRow:
    window_ms: int
    stride_ms: int
    latency_p50_ms: float
    latency_p95_ms: float
    event_accuracy: Optional[float]
    n_events: int
    mean_confidence: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_ms": int(self.window_ms),
            "stride_ms": int(self.stride_ms),
            "latency_p50_ms": round(float(self.latency_p50_ms), 3),
            "latency_p95_ms": round(float(self.latency_p95_ms), 3),
            "event_accuracy": None if self.event_accuracy is None else round(float(self.event_accuracy), 4),
            "n_events": int(self.n_events),
            "mean_confidence": None if self.mean_confidence is None else round(float(self.mean_confidence), 4),
        }


@dataclass
class WindowSweepReport:
    fs_hz: int
    channels: int
    rows: List[WindowSweepRow] = field(default_factory=list)
    recommended_window_ms: int = 600
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fs_hz": self.fs_hz,
            "channels": self.channels,
            "rows": [r.to_dict() for r in self.rows],
            "recommended_window_ms": int(self.recommended_window_ms),
            "notes": list(self.notes),
        }


def _eval_session_event_accuracy(
    *,
    model_path: str,
    session_dir: Path,
    window_ms: int,
    min_event_samples: int,
) -> tuple[Optional[float], int, Optional[float]]:
    session_dir = Path(session_dir)
    sig_path = session_dir / "signals.npy"
    ev_path = session_dir / "events.csv"
    if not sig_path.is_file() or not ev_path.is_file():
        return None, 0, None

    signals = np.load(sig_path)
    events = pd.read_csv(ev_path)
    if events.empty or "label" not in events.columns:
        return None, 0, None

    lm = load_model(str(model_path))
    emg = resolve_emg_mode_for_serve(checkpoint_emg_mode=lm.emg_mode, profile_preprocessing_mode=None)
    x = preprocess_streaming(
        signals.astype(np.float32),
        fs_hz=float(lm.fs),
        channels=int(lm.channels),
        mode=emg,
    )
    win = max(1, int(lm.fs * window_ms / 1000))
    correct = 0
    total = 0
    confs: List[float] = []

    for row in events.itertuples(index=False):
        label = str(getattr(row, "label", ""))
        if label not in lm.labels:
            continue
        start = int(getattr(row, "start_sample", 0))
        end = int(getattr(row, "end_sample", start))
        if end <= start:
            continue
        seg_len = end - start
        if seg_len < min_event_samples:
            center = (start + end) // 2
            half = win // 2
            s0 = max(0, center - half)
            s1 = min(x.shape[0], s0 + win)
            s0 = max(0, s1 - win)
        else:
            s1 = min(x.shape[0], end)
            s0 = max(0, s1 - win)
        if s1 - s0 < max(4, win // 4):
            continue
        seg = x[s0:s1, :]
        if seg.shape[0] < win:
            pad = np.zeros((win - seg.shape[0], seg.shape[1]), dtype=np.float32)
            seg = np.vstack([pad, seg])
        tok, conf = predict_preprocessed(lm, seg[-win:, :])
        total += 1
        confs.append(float(conf))
        if tok == label:
            correct += 1

    if total == 0:
        return None, 0, None
    acc = float(correct) / float(total)
    mean_conf = float(np.mean(confs)) if confs else None
    return acc, total, mean_conf


def run_window_sweep(
    *,
    model_path: str,
    session_dir: Optional[Path] = None,
    window_values_ms: Optional[Sequence[int]] = None,
    stride_ms: int = 120,
    n_latency_chunks: int = 80,
    target_latency_p95_ms: float = 500.0,
    min_accuracy: float = 0.0,
) -> WindowSweepReport:
    """Sweep inference windows; optional event accuracy if ``session_dir`` has labels."""
    lm = load_model(str(model_path))
    window_values_ms = list(window_values_ms or [400, 600, 900, 1200, 1500])
    report = WindowSweepReport(fs_hz=int(lm.fs), channels=int(lm.channels))

    best_score = -1.0
    best_ms = int(window_values_ms[0])
    min_ev = max(8, int(lm.fs * 0.05))

    for w_ms in window_values_ms:
        lat = run_latency_benchmark(
            model_path=str(model_path),
            window_ms=int(w_ms),
            stride_ms=int(stride_ms),
            n_chunks=int(n_latency_chunks),
            warmup_chunks=5,
        )
        acc, n_ev, mean_conf = (None, 0, None)
        if session_dir is not None:
            acc, n_ev, mean_conf = _eval_session_event_accuracy(
                model_path=str(model_path),
                session_dir=Path(session_dir),
                window_ms=int(w_ms),
                min_event_samples=min_ev,
            )
        row = WindowSweepRow(
            window_ms=int(w_ms),
            stride_ms=int(stride_ms),
            latency_p50_ms=float(lat.preprocess.p50_ms + lat.inference_window.p50_ms),
            latency_p95_ms=float(lat.preprocess.p95_ms + lat.inference_window.p95_ms),
            event_accuracy=acc,
            n_events=int(n_ev),
            mean_confidence=mean_conf,
        )
        report.rows.append(row)

        latency_ok = row.latency_p95_ms <= float(target_latency_p95_ms)
        acc_ok = acc is None or acc >= float(min_accuracy)
        score = (acc if acc is not None else 0.5) - (row.latency_p95_ms / 2000.0)
        if latency_ok and acc_ok and score > best_score:
            best_score = score
            best_ms = int(w_ms)

    report.recommended_window_ms = int(best_ms)
    report.notes.append(
        f"recommended window_ms={best_ms} (latency target {target_latency_p95_ms:.0f} ms p95)"
    )
    if session_dir is None:
        report.notes.append("pass --session for event-level accuracy on labeled data")
    return report
