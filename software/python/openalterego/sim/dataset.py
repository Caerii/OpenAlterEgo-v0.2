"""Synthetic dataset generation utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .stream import SimStream, SimStreamConfig


@dataclass
class DatasetConfig:
    out_dir: Path
    duration_s: float = 60.0
    config: SimStreamConfig = field(default_factory=SimStreamConfig)
    # If set, drop events shorter than this (helps avoid ultra-short segments)
    min_event_s: float = 0.15

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)


def generate_dataset(ds: DatasetConfig) -> Path:
    """Generate a session-like folder with signals.npy + events.csv."""
    ds.out_dir.mkdir(parents=True, exist_ok=True)

    sim = SimStream(ds.config)
    total_samples = int(ds.duration_s * ds.config.fs_hz)

    signals = np.zeros((total_samples, ds.config.channels), dtype=np.float32)
    i = 0
    while i < total_samples:
        chunk = sim.next_chunk()
        x = chunk.samples
        n = x.shape[0]
        take = min(n, total_samples - i)
        signals[i : i + take, :] = x[:take, :]
        i += take

    rows = []
    for ev in sim.events:
        dur_s = (ev.end_sample - ev.start_sample) / float(ds.config.fs_hz)
        if dur_s < ds.min_event_s:
            continue
        rows.append({"start_sample": int(ev.start_sample), "end_sample": int(ev.end_sample), "label": str(ev.label)})

    events = pd.DataFrame(rows, columns=["start_sample", "end_sample", "label"])
    signals_path = ds.out_dir / "signals.npy"
    events_path = ds.out_dir / "events.csv"

    np.save(signals_path, signals)
    events.to_csv(events_path, index=False)

    meta = {
        "fs_hz": int(ds.config.fs_hz),
        "channels": int(ds.config.channels),
        "duration_s": float(ds.duration_s),
        "labels": list(ds.config.scenario.labels),
        "seed": int(ds.config.seed),
        "sim_config": {
            "noise_uV": float(ds.config.noise_uV),
            "drift_uV_per_s": float(ds.config.drift_uV_per_s),
            "crosstalk": float(ds.config.crosstalk),
        },
    }
    (ds.out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return ds.out_dir
