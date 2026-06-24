"""Empirical training throughput benchmarks and bottleneck analysis."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from ..dsp.emg_config import EmgMode
from ..dsp.preprocess_cache import ensure_preprocessed_signals
from .data_split import resolve_train_val_indices
from .device import configure_cuda_for_training, resolve_device
from .model import create_model, default_arch
from .segment_cache import ensure_segment_arrays
from .train import SegmentDataset, fit_epochs
from .training_perf import (
    EpochTiming,
    TrainPhaseTiming,
    build_train_dataloader,
    default_num_workers,
    maybe_compile_model,
    resolve_use_amp,
)


def _load_session(data_dir: Path, *, mmap: bool) -> tuple[np.ndarray, pd.DataFrame]:
    sig_path = data_dir / "signals.npy"
    if mmap:
        signals = np.load(sig_path, mmap_mode="r")
    else:
        signals = np.load(sig_path)
    events = pd.read_csv(data_dir / "events.csv")
    return signals, events


def run_train_benchmark(
    data_dir: Path,
    *,
    fs_hz: int,
    segment_ms: int = 600,
    preprocess_mode: str = "streaming",
    emg_mode: EmgMode = "standard",
    arch: str = default_arch(),
    batch_size: int = 32,
    epochs: int = 1,
    device_preferred: str = "auto",
    no_amp: bool = False,
    num_workers: int = -1,
    compile_model: bool = False,
    mmap_signals: bool = False,
    rebuild_preprocess_cache: bool = False,
    rebuild_segment_cache: bool = False,
    val_split: float = 0.2,
    seed: int = 1337,
    split_by: str = "auto",
) -> TrainPhaseTiming:
    """Time setup phases + training epochs; return structured breakdown."""
    device = resolve_device(device_preferred)
    configure_cuda_for_training(device)
    use_amp = resolve_use_amp(device, no_amp=no_amp)
    nw = default_num_workers(int(num_workers))

    report = TrainPhaseTiming(
        device=str(device.type),
        use_amp=bool(use_amp),
        num_workers=int(nw),
        batch_size=int(batch_size),
    )

    t0 = time.perf_counter()
    signals, events = _load_session(data_dir, mmap=bool(mmap_signals))
    report.load_signals_s = time.perf_counter() - t0

    labels = sorted(events["label"].astype(str).unique().tolist())
    label_to_id = {lab: i for i, lab in enumerate(labels)}
    groups = events["trial_id"].astype(str).values if "trial_id" in events.columns else None
    tr_idx, val_idx = resolve_train_val_indices(
        events["label"].astype(str).values,
        float(val_split),
        int(seed),
        split_by=str(split_by),
        groups=groups,
    )
    tr_events = events.iloc[tr_idx].reset_index(drop=True)
    val_events = events.iloc[val_idx].reset_index(drop=True)
    report.n_train = int(len(tr_events))
    report.n_val = int(len(val_events))

    pp_mode = str(preprocess_mode)
    prepared = signals
    if pp_mode != "none":
        t_pp = time.perf_counter()
        prepared, cache_hit = ensure_preprocessed_signals(
            signals,
            data_dir,
            preprocess_mode=preprocess_mode,  # type: ignore[arg-type]
            emg_mode=emg_mode,
            fs_hz=int(fs_hz),
            use_cache=True,
            rebuild=bool(rebuild_preprocess_cache),
            show_progress=False,
        )
        report.preprocess_s = time.perf_counter() - t_pp
        report.preprocess_cache_hit = bool(cache_hit)
    else:
        prepared = np.asarray(signals, dtype=np.float32)

    t_seg = time.perf_counter()
    _, _, seg_tr_hit = ensure_segment_arrays(
        np.asarray(prepared, dtype=np.float32),
        tr_events,
        label_to_id,
        data_dir,
        preprocess_mode=pp_mode,
        emg_mode=emg_mode,
        fs_hz=int(fs_hz),
        segment_ms=int(segment_ms),
        seed=int(seed),
        split_tag="tr",
        use_cache=True,
        rebuild=bool(rebuild_segment_cache),
    )
    report.segment_train_s = time.perf_counter() - t_seg
    report.segment_train_cache_hit = bool(seg_tr_hit)

    t_seg = time.perf_counter()
    _, _, seg_val_hit = ensure_segment_arrays(
        np.asarray(prepared, dtype=np.float32),
        val_events,
        label_to_id,
        data_dir,
        preprocess_mode=pp_mode,
        emg_mode=emg_mode,
        fs_hz=int(fs_hz),
        segment_ms=int(segment_ms),
        seed=int(seed) + 1,
        split_tag="val",
        use_cache=True,
        rebuild=bool(rebuild_segment_cache),
    )
    report.segment_val_s = time.perf_counter() - t_seg
    report.segment_val_cache_hit = bool(seg_val_hit)

    t_dl = time.perf_counter()
    ds_tr = SegmentDataset(
        prepared,
        tr_events,
        label_to_id,
        fs_hz=int(fs_hz),
        segment_ms=int(segment_ms),
        preprocess_mode="none",
        emg_mode=emg_mode,
        seed=int(seed),
        session_dir=data_dir,
        split_tag="tr",
        use_segment_cache=True,
        rebuild_segment_cache=False,
        cache_preprocess_mode=pp_mode,
        cache_emg_mode=str(emg_mode),
    )
    ds_val = SegmentDataset(
        prepared,
        val_events,
        label_to_id,
        fs_hz=int(fs_hz),
        segment_ms=int(segment_ms),
        preprocess_mode="none",
        emg_mode=emg_mode,
        seed=int(seed) + 1,
        session_dir=data_dir,
        split_tag="val",
        use_segment_cache=True,
        rebuild_segment_cache=False,
        cache_preprocess_mode=pp_mode,
        cache_emg_mode=str(emg_mode),
    )
    dl_tr = build_train_dataloader(ds_tr, batch_size=int(batch_size), device=device, num_workers=nw)
    dl_val = build_train_dataloader(ds_val, batch_size=int(batch_size), device=device, num_workers=nw, shuffle=False)
    report.dataloader_init_s = time.perf_counter() - t_dl

    t_model = time.perf_counter()
    n_ch = int(ds_tr.X.shape[1])
    model = create_model(str(arch), channels=n_ch, classes=len(labels)).to(device)
    model = maybe_compile_model(model, enabled=bool(compile_model))
    report.model_init_s = time.perf_counter() - t_model

    epoch_profiles: List[EpochTiming] = []
    fit_epochs(
        model,
        dl_tr,
        dl_val,
        device,
        epochs=int(epochs),
        lr=1e-3,
        show_progress=False,
        save_best_path=None,
        use_amp=bool(use_amp),
        epoch_profiles=epoch_profiles,
    )
    report.epoch_timings = epoch_profiles
    return report


def compare_scenarios(
    data_dir: Path,
    *,
    fs_hz: int,
    segment_ms: int,
    preprocess_mode: str,
    emg_mode: EmgMode,
    batch_size: int,
    device_preferred: str = "auto",
) -> List[Dict[str, Any]]:
    """Run warm-cache benchmark across worker counts (and AMP on CUDA)."""
    device = resolve_device(device_preferred)
    scenarios: List[Dict[str, Any]] = []

    worker_opts = [0, 2, default_num_workers(-1)]
    worker_opts = sorted(set(worker_opts))

    amp_opts: List[tuple[str, bool]] = [("amp_default", False)]
    if device.type == "cuda":
        amp_opts = [("amp_on", False), ("amp_off", True)]

    for amp_name, no_amp in amp_opts:
        for nw in worker_opts:
            name = f"{amp_name}_workers{nw}"
            rep = run_train_benchmark(
                data_dir,
                fs_hz=int(fs_hz),
                segment_ms=int(segment_ms),
                preprocess_mode=str(preprocess_mode),
                emg_mode=emg_mode,
                batch_size=int(batch_size),
                epochs=1,
                device_preferred=device_preferred,
                no_amp=bool(no_amp),
                num_workers=int(nw),
                mmap_signals=True,
                rebuild_preprocess_cache=False,
                rebuild_segment_cache=False,
            )
            row = {"scenario": name, **rep.to_dict()}
            scenarios.append(row)
    return scenarios


def format_report_text(rep: TrainPhaseTiming) -> str:
    lines = [
        "=== Training benchmark ===",
        f"device={rep.device} amp={rep.use_amp} workers={rep.num_workers} batch={rep.batch_size}",
        f"train/val events: {rep.n_train}/{rep.n_val}",
        "",
        "Setup:",
        f"  load_signals:    {rep.load_signals_s:7.2f}s",
        f"  preprocess:      {rep.preprocess_s:7.2f}s  (cache_hit={rep.preprocess_cache_hit})",
        f"  segment_train:   {rep.segment_train_s:7.2f}s  (cache_hit={rep.segment_train_cache_hit})",
        f"  segment_val:     {rep.segment_val_s:7.2f}s  (cache_hit={rep.segment_val_cache_hit})",
        f"  dataloader_init: {rep.dataloader_init_s:7.2f}s",
        f"  model_init:      {rep.model_init_s:7.2f}s",
    ]
    if rep.epoch_timings:
        ep = rep.epoch_timings[-1]
        total = ep.train_s + ep.val_s
        lines.extend(
            [
                "",
                f"Last epoch (train {ep.train_s:.2f}s + val {ep.val_s:.2f}s = {total:.2f}s):",
                f"  dataloader wait: {ep.data_wait_s:7.2f}s  ({100 * ep.data_wait_s / max(total, 1e-9):.1f}%)",
                f"  compute:         {ep.compute_s:7.2f}s  ({100 * ep.compute_s / max(total, 1e-9):.1f}%)",
                f"  validation:      {ep.val_s:7.2f}s  ({100 * ep.val_s / max(total, 1e-9):.1f}%)",
                "",
                f"Bottleneck: {rep.bottleneck_label()}",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark training throughput and bottlenecks")
    ap.add_argument("--data", type=str, required=True, help="Session folder")
    ap.add_argument("--fs", type=int, required=True)
    ap.add_argument("--segment-ms", type=int, default=600)
    ap.add_argument("--preprocess-mode", type=str, default="streaming", choices=["offline", "streaming", "none"])
    ap.add_argument("--emg-mode", type=str, default="standard", choices=["standard", "clinical", "wide"])
    ap.add_argument("--arch", type=str, default=default_arch(), choices=["cnn", "se_resnet"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--num-workers", type=int, default=-1)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--mmap-signals", action="store_true", default=True)
    ap.add_argument("--no-mmap-signals", action="store_false", dest="mmap_signals")
    ap.add_argument("--rebuild-preprocess-cache", action="store_true")
    ap.add_argument("--rebuild-segment-cache", action="store_true")
    ap.add_argument("--compare", action="store_true", help="Sweep workers (and AMP on CUDA)")
    ap.add_argument("--out", type=str, default="", help="Write JSON report to this path")
    args = ap.parse_args()

    data_dir = Path(args.data)
    if args.compare:
        rows = compare_scenarios(
            data_dir,
            fs_hz=int(args.fs),
            segment_ms=int(args.segment_ms),
            preprocess_mode=str(args.preprocess_mode),
            emg_mode=args.emg_mode,  # type: ignore[arg-type]
            batch_size=int(args.batch_size),
            device_preferred=str(args.device),
        )
        print("=== Scenario comparison (warm caches) ===")
        for row in rows:
            ep_train = row.get("last_epoch_train_s", 0.0)
            ep_val = row.get("last_epoch_val_s", 0.0)
            print(
                f"{row['scenario']:24s}  epoch={ep_train + ep_val:6.1f}s  "
                f"bottleneck={row.get('bottleneck', '?')}"
            )
        payload = {"scenarios": rows}
        out_path = Path(args.out) if args.out else data_dir / "train_benchmark_compare.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote {out_path}")
        return

    cold = run_train_benchmark(
        data_dir,
        fs_hz=int(args.fs),
        segment_ms=int(args.segment_ms),
        preprocess_mode=str(args.preprocess_mode),
        emg_mode=args.emg_mode,  # type: ignore[arg-type]
        arch=str(args.arch),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        device_preferred=str(args.device),
        no_amp=bool(args.no_amp),
        num_workers=int(args.num_workers),
        compile_model=bool(args.compile),
        mmap_signals=bool(args.mmap_signals),
        rebuild_preprocess_cache=bool(args.rebuild_preprocess_cache),
        rebuild_segment_cache=bool(args.rebuild_segment_cache),
    )
    print(format_report_text(cold))

    warm = run_train_benchmark(
        data_dir,
        fs_hz=int(args.fs),
        segment_ms=int(args.segment_ms),
        preprocess_mode=str(args.preprocess_mode),
        emg_mode=args.emg_mode,  # type: ignore[arg-type]
        arch=str(args.arch),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        device_preferred=str(args.device),
        no_amp=bool(args.no_amp),
        num_workers=int(args.num_workers),
        compile_model=bool(args.compile),
        mmap_signals=bool(args.mmap_signals),
        rebuild_preprocess_cache=False,
        rebuild_segment_cache=False,
    )
    print("\n=== Warm cache rerun ===")
    print(format_report_text(warm))

    payload = {"cold": cold.to_dict(), "warm": warm.to_dict()}
    out_path = Path(args.out) if args.out else data_dir / "train_benchmark.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
