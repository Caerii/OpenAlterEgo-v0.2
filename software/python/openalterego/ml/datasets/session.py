"""OpenAlterEgo session folder writer (signals.npy + events.csv + meta.json)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


@dataclass
class SessionMeta:
    fs_hz: float
    channels: int
    source: str
    notes: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "fs_hz": float(self.fs_hz),
            "channels": int(self.channels),
            "source": str(self.source),
        }
        if self.notes:
            d["notes"] = list(self.notes)
        d.update(self.extra)
        return d


def write_session_folder(
    out_dir: Union[str, Path],
    signals: np.ndarray,
    events: pd.DataFrame,
    meta: SessionMeta,
) -> Path:
    """Write a training-ready session directory."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sig = np.asarray(signals, dtype=np.float32)
    if sig.ndim != 2:
        raise ValueError(f"signals must be 2D (time, channels), got {sig.shape}")
    ev = events.copy()
    for col in ("start_sample", "end_sample", "label"):
        if col not in ev.columns:
            raise ValueError(f"events missing column {col!r}")
    np.save(out / "signals.npy", sig)
    ev.to_csv(out / "events.csv", index=False)
    (out / "meta.json").write_text(json.dumps(meta.to_dict(), indent=2), encoding="utf-8")
    return out
