"""Phase-1 Gowda preprocessing ablations (scientific matrix)."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..data_split import gowda_sentence_train_val_indices
from ..device import configure_cuda_for_training, resolve_device
from ..eval.metrics import split_metrics
from ..model import create_model, default_arch
from ..train import SegmentDataset, fit_epochs
from ..training_perf import build_train_dataloader, maybe_compile_model, resolve_use_amp


@dataclass(frozen=True)
class AblationConfig:
    name: str
    preprocess_mode: str
    emg_mode: str
    segment_ms: int
    per_event_preprocess: bool

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "preprocess_mode": self.preprocess_mode,
            "emg_mode": self.emg_mode,
            "segment_ms": int(self.segment_ms),
            "per_event_preprocess": bool(self.per_event_preprocess),
        }


PHASE1_CONFIGS: List[AblationConfig] = [
    AblationConfig("baseline_stream_wide_900", "streaming", "wide", 900, False),
    AblationConfig("per_event_wide_900", "offline", "wide", 900, True),
    AblationConfig("per_event_gowda_900", "offline", "gowda", 900, True),
    AblationConfig("per_event_gowda_2000", "offline", "gowda", 2000, True),
]


@torch.no_grad()
def _predict_split(
    model: torch.nn.Module,
    dataset: SegmentDataset,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
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


def run_ablation(
    data_dir: Path,
    cfg: AblationConfig,
    *,
    fs_hz: int = 5000,
    epochs: int = 30,
    batch_size: int = 64,
    seed: int = 1337,
    device_preferred: str = "auto",
    arch: str = default_arch(),
    rebuild_segment_cache: bool = False,
) -> Dict[str, Any]:
    """Train one config; return train/val metrics."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = resolve_device(device_preferred)
    configure_cuda_for_training(device)
    use_amp = resolve_use_amp(device, no_amp=False)

    signals = np.load(data_dir / "signals.npy", mmap_mode="r")
    events = pd.read_csv(data_dir / "events.csv")
    labels = sorted({str(x) for x in events["label"].unique()})
    label_to_id = {lab: i for i, lab in enumerate(labels)}

    tr_idx, val_idx = gowda_sentence_train_val_indices(events["trial_id"].astype(int).values)
    tr_events = events.iloc[tr_idx].reset_index(drop=True)
    val_events = events.iloc[val_idx].reset_index(drop=True)

    pp_mode = "per_event" if cfg.per_event_preprocess else str(cfg.preprocess_mode)
    prepared = np.asarray(signals, dtype=np.float32)

    if not cfg.per_event_preprocess and cfg.preprocess_mode != "none":
        from ...dsp.preprocess_cache import ensure_preprocessed_signals

        prepared, _ = ensure_preprocessed_signals(
            signals,
            data_dir,
            preprocess_mode=cfg.preprocess_mode,  # type: ignore[arg-type]
            emg_mode=cfg.emg_mode,  # type: ignore[arg-type]
            fs_hz=int(fs_hz),
            use_cache=True,
            rebuild=False,
            show_progress=False,
        )

    ds_tr = SegmentDataset(
        prepared,
        tr_events,
        label_to_id,
        fs_hz=int(fs_hz),
        segment_ms=int(cfg.segment_ms),
        preprocess_mode="none",
        emg_mode=cfg.emg_mode,  # type: ignore[arg-type]
        seed=int(seed),
        session_dir=data_dir,
        split_tag=f"ab_{cfg.name}_tr",
        use_segment_cache=True,
        rebuild_segment_cache=bool(rebuild_segment_cache),
        cache_preprocess_mode=pp_mode,
        cache_emg_mode=str(cfg.emg_mode),
        per_event_preprocess=bool(cfg.per_event_preprocess),
    )
    ds_val = SegmentDataset(
        prepared,
        val_events,
        label_to_id,
        fs_hz=int(fs_hz),
        segment_ms=int(cfg.segment_ms),
        preprocess_mode="none",
        emg_mode=cfg.emg_mode,  # type: ignore[arg-type]
        seed=int(seed) + 1,
        session_dir=data_dir,
        split_tag=f"ab_{cfg.name}_val",
        use_segment_cache=True,
        rebuild_segment_cache=bool(rebuild_segment_cache),
        cache_preprocess_mode=pp_mode,
        cache_emg_mode=str(cfg.emg_mode),
        per_event_preprocess=bool(cfg.per_event_preprocess),
    )

    n_ch = int(ds_tr.X.shape[1])
    model = create_model(str(arch), channels=n_ch, classes=len(labels)).to(device)
    model = maybe_compile_model(model, enabled=False)
    dl_tr = build_train_dataloader(ds_tr, batch_size=int(batch_size), device=device, num_workers=0)
    dl_val = build_train_dataloader(ds_val, batch_size=int(batch_size), device=device, num_workers=0, shuffle=False)

    ckpt_path = data_dir / "ablations" / f"{cfg.name}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
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
            "emg_mode": str(cfg.emg_mode),
            "segment_ms": int(cfg.segment_ms),
            "per_event_preprocess": bool(cfg.per_event_preprocess),
            "arch": str(arch),
            "ablation": cfg.name,
        },
        use_amp=bool(use_amp),
    )
    elapsed = time.perf_counter() - t0

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])

    y_tr, p_tr = _predict_split(model, ds_tr, device, batch_size)
    y_va, p_va = _predict_split(model, ds_val, device, batch_size)
    train_m = split_metrics(y_tr, p_tr, labels=labels)
    val_m = split_metrics(y_va, p_va, labels=labels)

    return {
        "config": cfg.to_dict(),
        "device": str(device.type),
        "use_amp": bool(use_amp),
        "best_val_acc_checkpoint": round(float(best_val), 4),
        "train": train_m.to_dict(),
        "val": val_m.to_dict(),
        "train_val_gap": round(float(train_m.accuracy - val_m.accuracy), 4),
        "elapsed_s": round(float(elapsed), 1),
        "checkpoint": str(ckpt_path),
    }


def run_phase1_matrix(
    data_dir: Path,
    *,
    fs_hz: int = 5000,
    epochs: int = 30,
    batch_size: int = 64,
    device_preferred: str = "auto",
    configs: Optional[List[AblationConfig]] = None,
) -> Dict[str, Any]:
    configs = list(configs or PHASE1_CONFIGS)
    rows: List[Dict[str, Any]] = []
    for cfg in configs:
        print(f"[openalterego] ablation: {cfg.name}")
        rows.append(
            run_ablation(
                data_dir,
                cfg,
                fs_hz=int(fs_hz),
                epochs=int(epochs),
                batch_size=int(batch_size),
                device_preferred=device_preferred,
            )
        )
    best = max(rows, key=lambda r: float(r["val"]["macro_f1"]))
    return {
        "phase": "1_preprocess_window",
        "session": str(data_dir),
        "fs_hz": int(fs_hz),
        "epochs": int(epochs),
        "rows": rows,
        "best_by_val_macro_f1": best["config"]["name"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Gowda Phase-1 preprocessing ablation matrix")
    ap.add_argument("--data", type=str, required=True, help="Session folder (gowda_top30)")
    ap.add_argument("--fs", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--out", type=str, default="", help="JSON report path")
    ap.add_argument("--rebuild-segment-cache", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data)
    report = run_phase1_matrix(
        data_dir,
        fs_hz=int(args.fs),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        device_preferred=str(args.device),
    )
    out = Path(args.out) if args.out else data_dir / "ablations" / "phase1_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== Phase 1 ablation summary ===")
    print(f"{'config':28s}  train_acc  val_acc  val_F1  gap")
    for row in report["rows"]:
        c = row["config"]["name"]
        print(
            f"{c:28s}  {row['train']['accuracy']:8.3f}  "
            f"{row['val']['accuracy']:7.3f}  {row['val']['macro_f1']:6.3f}  {row['train_val_gap']:+.3f}"
        )
    print(f"\nBest (macro-F1): {report['best_by_val_macro_f1']}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
