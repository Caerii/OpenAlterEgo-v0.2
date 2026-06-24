"""Load and sanitize Gowda-style event tables."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def sanitize_trial_events(events: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing trial_id; cast trial columns to int."""
    if "trial_id" not in events.columns:
        return events
    out = events.dropna(subset=["trial_id"]).copy()
    out["trial_id"] = out["trial_id"].astype(int)
    if "word_idx" in out.columns:
        out["word_idx"] = out["word_idx"].fillna(0).astype(int)
    return out.reset_index(drop=True)


def load_gowda_events(data_dir: Path) -> pd.DataFrame:
    """Load ``events.csv`` with trial columns sanitized."""
    return sanitize_trial_events(pd.read_csv(Path(data_dir) / "events.csv"))


def read_split_mode(data_dir: Path) -> str:
    meta_path = Path(data_dir) / "meta.json"
    if not meta_path.is_file():
        return ""
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    return str(meta.get("split_mode", ""))
