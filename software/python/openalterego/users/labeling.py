"""Offline labeling helpers for BLE sessions (events.csv from markers)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import pandas as pd


def _require_columns(df: pd.DataFrame, cols: List[str], path: Path) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}; have {list(df.columns)}")


def markers_to_events(
    markers_path: Union[str, Path],
    *,
    fs_hz: float,
    min_duration_s: float = 0.12,
) -> pd.DataFrame:
    """Convert a marker file to ``events.csv`` columns ``start_sample,end_sample,label``.

    Supported marker formats (auto-detected):

    1. ``start_sample,end_sample,label``
    2. ``sample,label`` — point marker; duration = ``min_duration_s``
    3. ``time_s,label`` — point marker at ``time_s * fs``
    4. ``start_s,end_s,label`` — interval in seconds
    """
    path = Path(markers_path)
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{path}: empty markers file")

    cols = {c.strip().lower(): c for c in df.columns}
    fs = float(fs_hz)
    min_n = max(1, int(min_duration_s * fs))
    rows: List[dict] = []

    if "start_sample" in cols and "end_sample" in cols and "label" in cols:
        for r in df.itertuples(index=False):
            d = r._asdict()
            start = int(d[cols["start_sample"]])
            end = int(d[cols["end_sample"]])
            label = str(d[cols["label"]])
            if end > start:
                rows.append({"start_sample": start, "end_sample": end, "label": label})
        return pd.DataFrame(rows, columns=["start_sample", "end_sample", "label"])

    if "start_s" in cols and "end_s" in cols and "label" in cols:
        for r in df.itertuples(index=False):
            d = r._asdict()
            start = int(float(d[cols["start_s"]]) * fs)
            end = int(float(d[cols["end_s"]]) * fs)
            label = str(d[cols["label"]])
            if end > start:
                rows.append({"start_sample": start, "end_sample": end, "label": label})
        return pd.DataFrame(rows, columns=["start_sample", "end_sample", "label"])

    label_col = cols.get("label")
    if label_col is None:
        raise ValueError(f"{path}: need a 'label' column")

    if "sample" in cols:
        for r in df.itertuples(index=False):
            d = r._asdict()
            start = int(d[cols["sample"]])
            end = start + min_n
            rows.append({"start_sample": start, "end_sample": end, "label": str(d[label_col])})
        return pd.DataFrame(rows, columns=["start_sample", "end_sample", "label"])

    if "time_s" in cols:
        for r in df.itertuples(index=False):
            d = r._asdict()
            start = int(float(d[cols["time_s"]]) * fs)
            end = start + min_n
            rows.append({"start_sample": start, "end_sample": end, "label": str(d[label_col])})
        return pd.DataFrame(rows, columns=["start_sample", "end_sample", "label"])

    raise ValueError(
        f"{path}: unsupported columns {list(df.columns)}; "
        "use start_sample/end_sample/label, sample/label, time_s/label, or start_s/end_s/label"
    )


def write_events_csv(events: pd.DataFrame, session_dir: Union[str, Path]) -> Path:
    """Write ``events.csv`` into a session folder."""
    out = Path(session_dir) / "events.csv"
    events.to_csv(out, index=False)
    return out


def label_session_from_markers(
    session_dir: Union[str, Path],
    markers_path: Union[str, Path],
    *,
    fs_hz: Optional[float] = None,
    min_duration_s: float = 0.12,
) -> Path:
    """Import markers and write ``events.csv`` (reads ``fs_hz`` from ``meta.json`` if omitted)."""
    session_dir = Path(session_dir)
    fs = float(fs_hz) if fs_hz is not None else _fs_from_session(session_dir)
    events = markers_to_events(markers_path, fs_hz=fs, min_duration_s=float(min_duration_s))
    if events.empty:
        raise ValueError("no events produced from markers")
    return write_events_csv(events, session_dir)


def _fs_from_session(session_dir: Path) -> float:
    meta = session_dir / "meta.json"
    if meta.is_file():
        import json

        data = json.loads(meta.read_text(encoding="utf-8"))
        if "fs_hz" in data:
            return float(data["fs_hz"])
    sess = session_dir / "session.json"
    if sess.is_file():
        import json

        data = json.loads(sess.read_text(encoding="utf-8"))
        if "fs_hz" in data:
            return float(data["fs_hz"])
    raise ValueError(f"fs_hz required; not found in {session_dir}/meta.json or session.json")


def events_template_csv(labels: List[str], path: Union[str, Path]) -> Path:
    """Write an empty ``events.csv`` header plus comment rows for manual editing."""
    out = Path(path)
    lines = ["start_sample,end_sample,label"]
    for lab in labels:
        lines.append(f"# example: 0,250,{lab}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
