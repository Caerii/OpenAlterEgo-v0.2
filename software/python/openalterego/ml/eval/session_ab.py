"""Session-level preprocessing A/B and quick train eval on imported real data."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ...dsp.preprocess_cache import ensure_preprocessed_signals
from ..data_split import stratified_train_val_indices
from ..model import create_model, default_arch
from ..train import SegmentDataset, evaluate, fit_epochs
from ...dsp.quality import assess_signal_quality
from ...dsp.emg_config import validate_emg_wide_fs


@dataclass
class PreprocessABRow:
    emg_mode: str
    snr_db: Optional[float]
    motion_index: float
    train_acc: float
    val_acc: float
    val_loss: float
    n_train: int
    n_val: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "emg_mode": self.emg_mode,
            "snr_db": None if self.snr_db is None else round(float(self.snr_db), 3),
            "motion_index": round(float(self.motion_index), 4),
            "train_acc": round(float(self.train_acc), 4),
            "val_acc": round(float(self.val_acc), 4),
            "val_loss": round(float(self.val_loss), 4),
            "n_train": int(self.n_train),
            "n_val": int(self.n_val),
        }


@dataclass
class SessionABReport:
    session_dir: str
    fs_hz: int
    channels: int
    n_labels: int
    arch: str
    rows: List[PreprocessABRow] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_dir": self.session_dir,
            "fs_hz": int(self.fs_hz),
            "channels": int(self.channels),
            "n_labels": int(self.n_labels),
            "arch": self.arch,
            "rows": [r.to_dict() for r in self.rows],
            "notes": list(self.notes),
        }


def _load_session(session_dir: Path) -> tuple[np.ndarray, pd.DataFrame, int, dict]:
    session_dir = Path(session_dir)
    signals = np.load(session_dir / "signals.npy")
    events = pd.read_csv(session_dir / "events.csv")
    meta: dict = {}
    meta_path = session_dir / "meta.json"
    if meta_path.is_file():
        import json

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fs = int(meta.get("fs_hz", 250))
    return signals.astype(np.float32), events, fs, meta


def run_session_preprocess_ab(
    session_dir: Path,
    *,
    emg_modes: Optional[List[str]] = None,
    segment_ms: int = 600,
    epochs: int = 12,
    batch_size: int = 16,
    val_split: float = 0.2,
    seed: int = 1337,
    arch: str = default_arch(),
    min_samples_per_label: int = 2,
    show_progress: bool = False,
) -> SessionABReport:
    """Train/eval quick models under standard vs wide (etc.) on one session folder."""
    import torch
    from torch.utils.data import DataLoader

    signals, events, fs_hz, meta = _load_session(session_dir)
    channels = int(signals.shape[1])
    emg_modes = list(emg_modes or ["standard", "wide"])

    labels = sorted({str(x) for x in events["label"].unique()})
    counts = events["label"].astype(str).value_counts()
    keep = {lab for lab, n in counts.items() if int(n) >= int(min_samples_per_label)}
    events = events[events["label"].astype(str).isin(keep)].reset_index(drop=True)
    labels = sorted(keep)
    if len(labels) < 2:
        raise ValueError(
            f"need >=2 labels with >={min_samples_per_label} samples; "
            f"got labels={labels} counts={counts.to_dict()}"
        )

    label_to_id = {lab: i for i, lab in enumerate(labels)}
    tr_idx, val_idx = stratified_train_val_indices(
        events["label"].astype(str).values,
        float(val_split),
        int(seed),
    )
    tr_events = events.iloc[tr_idx].reset_index(drop=True)
    val_events = events.iloc[val_idx].reset_index(drop=True)
    if len(val_events) == 0 and len(tr_events) > 1:
        val_events = tr_events.iloc[-1:].reset_index(drop=True)
        tr_events = tr_events.iloc[:-1].reset_index(drop=True)

    report = SessionABReport(
        session_dir=str(session_dir),
        fs_hz=int(fs_hz),
        channels=int(channels),
        n_labels=len(labels),
        arch=str(arch),
    )
    if meta.get("source"):
        report.notes.append(f"source={meta['source']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for mode in emg_modes:
        m = str(mode)
        if m == "wide":
            validate_emg_wide_fs(float(fs_hz))
        if m == "wide":
            band = (20.0, 450.0)
        elif m == "clinical":
            band = (0.5, 8.0)
        else:
            band = (1.0, 50.0)
        q = assess_signal_quality(
            signals,
            fs_hz=float(fs_hz),
            signal_band_hz=band,
            noise_band_hz=(0.5, 5.0),
            low_freq_cutoff_hz=5.0,
            axis=0,
        )
        prepared, cache_hit = ensure_preprocessed_signals(
            signals,
            session_dir,
            preprocess_mode="streaming",
            emg_mode=m,  # type: ignore[arg-type]
            fs_hz=int(fs_hz),
            use_cache=True,
            rebuild=False,
            show_progress=show_progress and not cache_hit,
        )
        if show_progress:
            note = "cache hit" if cache_hit else "computed"
            print(f"[openalterego] A/B {m}: preprocess {note}")
        ds_tr = SegmentDataset(
            prepared,
            tr_events,
            label_to_id,
            fs_hz=int(fs_hz),
            segment_ms=int(segment_ms),
            preprocess_mode="none",
            emg_mode=m,  # type: ignore[arg-type]
            seed=int(seed),
        )
        ds_val = SegmentDataset(
            prepared,
            val_events,
            label_to_id,
            fs_hz=int(fs_hz),
            segment_ms=int(segment_ms),
            preprocess_mode="none",
            emg_mode=m,  # type: ignore[arg-type]
            seed=int(seed) + 1,
        )
        if len(ds_tr) == 0 or len(ds_val) == 0:
            report.notes.append(f"skip {m}: empty train/val segments")
            continue

        model = create_model(str(arch), channels=channels, classes=len(labels)).to(device)
        dl_tr = DataLoader(ds_tr, batch_size=int(batch_size), shuffle=True)
        dl_val = DataLoader(ds_val, batch_size=int(batch_size), shuffle=False)
        _, val_loss, val_acc = fit_epochs(
            model,
            dl_tr,
            dl_val,
            device,
            epochs=int(epochs),
            lr=1e-3,
            show_progress=show_progress,
        )
        # train acc on final epoch weights
        model.eval()
        tr_correct = 0
        tr_n = 0
        with torch.no_grad():
            for x, y in dl_tr:
                x = x.to(device)
                y = y.to(device)
                pred = model(x).argmax(dim=1)
                tr_correct += int((pred == y).sum().item())
                tr_n += int(y.size(0))
        train_acc = float(tr_correct) / max(tr_n, 1)
        report.rows.append(
            PreprocessABRow(
                emg_mode=m,
                snr_db=q.snr_db,
                motion_index=float(q.motion_index),
                train_acc=train_acc,
                val_acc=float(val_acc),
                val_loss=float(val_loss),
                n_train=len(ds_tr),
                n_val=len(ds_val),
            )
        )

    if not report.rows:
        raise ValueError("no A/B rows produced")
    best = max(report.rows, key=lambda r: r.val_acc)
    report.notes.append(f"best val_acc={best.val_acc:.3f} emg_mode={best.emg_mode}")
    return report
