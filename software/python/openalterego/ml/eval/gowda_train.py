"""Shared Gowda word-classifier training for ablations and Phase 2."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from ..data_split import gowda_sentence_train_val_indices
from ..device import configure_cuda_for_training, resolve_device
from ..eval.metrics import split_metrics
from ..model import create_model, default_arch
from ..train import SegmentDataset, fit_epochs
from ..training_perf import build_train_dataloader, maybe_compile_model, resolve_use_amp

WINNING_CONFIG = {
    "name": "per_event_gowda_2000",
    "preprocess_mode": "offline",
    "emg_mode": "gowda",
    "segment_ms": 2000,
    "per_event_preprocess": True,
}

WEEKDAY_LABELS = frozenset(
    {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}
)


def filter_events(
    events: pd.DataFrame,
    *,
    labels: Optional[frozenset[str]] = None,
    word_idx: Optional[int] = None,
) -> pd.DataFrame:
    out = events.copy()
    if labels is not None:
        out = out[out["label"].astype(str).isin(labels)]
    if word_idx is not None and "word_idx" in out.columns:
        out = out[out["word_idx"].astype(int) == int(word_idx)]
    return out.reset_index(drop=True)


@torch.no_grad()
def predict_dataset(
    model: torch.nn.Module,
    dataset: SegmentDataset,
    device: torch.device,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    dl = build_train_dataloader(dataset, batch_size=int(batch_size), device=device, num_workers=0, shuffle=False)
    preds: List[int] = []
    ys: List[int] = []
    model.eval()
    for x, y in dl:
        x = x.to(device, non_blocking=device.type == "cuda")
        logits = model(x)
        preds.extend(logits.argmax(dim=1).cpu().tolist())
        ys.extend(y.cpu().tolist())
    return np.asarray(ys, dtype=np.int64), np.asarray(preds, dtype=np.int64)


def eval_checkpoint(
    data_dir: Path,
    ckpt_path: Path,
    *,
    events_filter: Optional[pd.DataFrame] = None,
    batch_size: int = 64,
    device_preferred: str = "auto",
) -> Dict[str, Any]:
    """Load checkpoint and compute train/val metrics (no training)."""
    device = resolve_device(device_preferred)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    labels: List[str] = list(ckpt["labels"])
    label_to_id = {lab: i for i, lab in enumerate(labels)}
    fs_hz = int(ckpt.get("fs", 5000))
    segment_ms = int(ckpt.get("segment_ms", 2000))
    emg_mode = str(ckpt.get("emg_mode", "gowda"))
    per_event = bool(ckpt.get("per_event_preprocess", True))
    pp_mode = str(ckpt.get("preprocess_mode", "per_event"))
    seed = int(ckpt.get("seed", 1337))
    arch = str(ckpt.get("arch", default_arch()))
    n_ch = int(ckpt["channels"])

    signals = np.load(data_dir / "signals.npy", mmap_mode="r")
    events = pd.read_csv(data_dir / "events.csv")
    if events_filter is not None:
        keys = events_filter[["trial_id", "start_sample", "end_sample", "label"]].drop_duplicates()
        events = events.merge(keys, on=["trial_id", "start_sample", "end_sample", "label"], how="inner")
        events = events.reset_index(drop=True)

    tr_idx, val_idx = gowda_sentence_train_val_indices(events["trial_id"].astype(int).values)
    tr_events = events.iloc[tr_idx].reset_index(drop=True)
    val_events = events.iloc[val_idx].reset_index(drop=True)
    prepared = np.asarray(signals, dtype=np.float32)

    tag = f"eval_{ckpt_path.stem}"
    ds_tr = SegmentDataset(
        prepared, tr_events, label_to_id, fs_hz=fs_hz, segment_ms=segment_ms,
        preprocess_mode="none", emg_mode=emg_mode,  # type: ignore[arg-type]
        seed=seed, session_dir=data_dir, split_tag=f"{tag}_tr",
        cache_preprocess_mode=pp_mode, cache_emg_mode=emg_mode,
        per_event_preprocess=per_event,
    )
    ds_val = SegmentDataset(
        prepared, val_events, label_to_id, fs_hz=fs_hz, segment_ms=segment_ms,
        preprocess_mode="none", emg_mode=emg_mode,  # type: ignore[arg-type]
        seed=seed + 1, session_dir=data_dir, split_tag=f"{tag}_val",
        cache_preprocess_mode=pp_mode, cache_emg_mode=emg_mode,
        per_event_preprocess=per_event,
    )
    model = create_model(arch, channels=n_ch, classes=len(labels)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    y_tr, p_tr = predict_dataset(model, ds_tr, device, batch_size)
    y_va, p_va = predict_dataset(model, ds_val, device, batch_size)
    train_m = split_metrics(y_tr, p_tr, labels=labels)
    val_m = split_metrics(y_va, p_va, labels=labels)
    return {
        "seed": seed,
        "n_classes": len(labels),
        "n_train": len(tr_events),
        "n_val": len(val_events),
        "labels": labels,
        "train": train_m.to_dict(),
        "val": val_m.to_dict(),
        "train_val_gap": round(float(train_m.accuracy - val_m.accuracy), 4),
        "checkpoint": str(ckpt_path),
        "y_val": y_va.tolist(),
        "p_val": p_va.tolist(),
    }


def train_word_classifier(
    data_dir: Path,
    *,
    fs_hz: int = 5000,
    emg_mode: str = "gowda",
    segment_ms: int = 2000,
    per_event_preprocess: bool = True,
    preprocess_mode: str = "offline",
    epochs: int = 30,
    batch_size: int = 64,
    seed: int = 1337,
    device_preferred: str = "auto",
    arch: str = default_arch(),
    events_filter: Optional[pd.DataFrame] = None,
    split_tag_prefix: str = "run",
    save_path: Optional[Path] = None,
    rebuild_segment_cache: bool = False,
) -> Dict[str, Any]:
    """Train SE-ResNet word classifier; return metrics + model path."""
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    device = resolve_device(device_preferred)
    configure_cuda_for_training(device)
    use_amp = resolve_use_amp(device, no_amp=False)

    signals = np.load(data_dir / "signals.npy", mmap_mode="r")
    events = pd.read_csv(data_dir / "events.csv")
    if events_filter is not None:
        keys = events_filter[["trial_id", "start_sample", "end_sample", "label"]].drop_duplicates()
        events = events.merge(keys, on=["trial_id", "start_sample", "end_sample", "label"], how="inner")
        events = events.reset_index(drop=True)

    labels = sorted({str(x) for x in events["label"].unique()})
    if len(labels) < 2:
        raise ValueError(f"Need >= 2 classes after filter; got {labels}")
    label_to_id = {lab: i for i, lab in enumerate(labels)}

    tr_idx, val_idx = gowda_sentence_train_val_indices(events["trial_id"].astype(int).values)
    tr_events = events.iloc[tr_idx].reset_index(drop=True)
    val_events = events.iloc[val_idx].reset_index(drop=True)

    pp_mode = "per_event" if per_event_preprocess else str(preprocess_mode)
    prepared = np.asarray(signals, dtype=np.float32)

    if not per_event_preprocess and preprocess_mode != "none":
        from ...dsp.preprocess_cache import ensure_preprocessed_signals

        prepared, _ = ensure_preprocessed_signals(
            signals,
            data_dir,
            preprocess_mode=preprocess_mode,  # type: ignore[arg-type]
            emg_mode=emg_mode,  # type: ignore[arg-type]
            fs_hz=int(fs_hz),
            use_cache=True,
            rebuild=False,
            show_progress=False,
        )

    tag = f"{split_tag_prefix}_s{seed}"
    ds_tr = SegmentDataset(
        prepared,
        tr_events,
        label_to_id,
        fs_hz=int(fs_hz),
        segment_ms=int(segment_ms),
        preprocess_mode="none",
        emg_mode=emg_mode,  # type: ignore[arg-type]
        seed=int(seed),
        session_dir=data_dir,
        split_tag=f"{tag}_tr",
        use_segment_cache=True,
        rebuild_segment_cache=bool(rebuild_segment_cache),
        cache_preprocess_mode=pp_mode,
        cache_emg_mode=str(emg_mode),
        per_event_preprocess=bool(per_event_preprocess),
    )
    ds_val = SegmentDataset(
        prepared,
        val_events,
        label_to_id,
        fs_hz=int(fs_hz),
        segment_ms=int(segment_ms),
        preprocess_mode="none",
        emg_mode=emg_mode,  # type: ignore[arg-type]
        seed=int(seed) + 1,
        session_dir=data_dir,
        split_tag=f"{tag}_val",
        use_segment_cache=True,
        rebuild_segment_cache=bool(rebuild_segment_cache),
        cache_preprocess_mode=pp_mode,
        cache_emg_mode=str(emg_mode),
        per_event_preprocess=bool(per_event_preprocess),
    )

    n_ch = int(ds_tr.X.shape[1])
    model = create_model(str(arch), channels=n_ch, classes=len(labels)).to(device)
    model = maybe_compile_model(model, enabled=False)
    dl_tr = build_train_dataloader(ds_tr, batch_size=int(batch_size), device=device, num_workers=0)
    dl_val = build_train_dataloader(ds_val, batch_size=int(batch_size), device=device, num_workers=0, shuffle=False)

    ckpt_path = save_path or (data_dir / "ablations" / f"{split_tag_prefix}_seed{seed}.pt")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    best_val, _, _ = fit_epochs(
        model,
        dl_tr,
        dl_val,
        device,
        epochs=int(epochs),
        lr=1e-3,
        show_progress=False,
        save_best_path=ckpt_path,
        extra_ckpt={
            "labels": labels,
            "fs": int(fs_hz),
            "channels": n_ch,
            "preprocess_mode": pp_mode,
            "emg_mode": str(emg_mode),
            "segment_ms": int(segment_ms),
            "per_event_preprocess": bool(per_event_preprocess),
            "arch": str(arch),
            "seed": int(seed),
        },
        use_amp=bool(use_amp),
    )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    y_tr, p_tr = predict_dataset(model, ds_tr, device, batch_size)
    y_va, p_va = predict_dataset(model, ds_val, device, batch_size)
    train_m = split_metrics(y_tr, p_tr, labels=labels)
    val_m = split_metrics(y_va, p_va, labels=labels)

    return {
        "seed": int(seed),
        "n_classes": len(labels),
        "n_train": int(len(tr_events)),
        "n_val": int(len(val_events)),
        "labels": labels,
        "best_val_acc_checkpoint": round(float(best_val), 4),
        "train": train_m.to_dict(),
        "val": val_m.to_dict(),
        "train_val_gap": round(float(train_m.accuracy - val_m.accuracy), 4),
        "checkpoint": str(ckpt_path),
        "y_val": y_va.tolist(),
        "p_val": p_va.tolist(),
    }
