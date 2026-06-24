"""Phase 2: weekday ceiling, multi-seed bootstrap, CTC phoneme path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from ..ctc.train import train_gowda_ctc
from .bootstrap import summarize_multiseed
from .gowda_train import WEEKDAY_LABELS, filter_events, train_word_classifier


def run_weekday_ceiling(
    data_dir: Path,
    *,
    fs_hz: int = 5000,
    epochs: int = 30,
    batch_size: int = 64,
    seed: int = 1337,
    device_preferred: str = "auto",
) -> Dict[str, Any]:
    """7-way weekday classification (slot 0 only) with winning preprocess config."""
    events = pd.read_csv(data_dir / "events.csv")
    filtered = filter_events(events, labels=WEEKDAY_LABELS, word_idx=0)
    row = train_word_classifier(
        data_dir,
        fs_hz=int(fs_hz),
        emg_mode="gowda",
        segment_ms=2000,
        per_event_preprocess=True,
        epochs=int(epochs),
        batch_size=int(batch_size),
        seed=int(seed),
        device_preferred=device_preferred,
        events_filter=filtered,
        split_tag_prefix="weekday7",
        save_path=data_dir / "ablations" / "weekday7_ceiling.pt",
    )
    return {
        "task": "weekday7_slot0",
        "n_classes": 7,
        "chance": round(1.0 / 7.0, 4),
        **{k: v for k, v in row.items() if k not in ("y_val", "p_val")},
    }


def run_multiseed_bootstrap(
    data_dir: Path,
    *,
    fs_hz: int = 5000,
    epochs: int = 30,
    batch_size: int = 64,
    seeds: List[int],
    device_preferred: str = "auto",
    n_bootstrap: int = 2000,
) -> Dict[str, Any]:
    """Multi-seed training on winning word-classifier config + bootstrap CI."""
    runs = []
    for seed in seeds:
        print(f"[openalterego] multiseed seed={seed}")
        runs.append(
            train_word_classifier(
                data_dir,
                fs_hz=int(fs_hz),
                emg_mode="gowda",
                segment_ms=2000,
                per_event_preprocess=True,
                epochs=int(epochs),
                batch_size=int(batch_size),
                seed=int(seed),
                device_preferred=device_preferred,
                split_tag_prefix="multiseed",
                save_path=data_dir / "ablations" / f"multiseed_seed{seed}.pt",
            )
        )
    summary = summarize_multiseed(runs, n_bootstrap=int(n_bootstrap))
    return {
        "task": "multiseed_bootstrap",
        "config": "per_event_gowda_2000",
        **summary,
    }


def run_ctc_path(
    data_dir: Path,
    *,
    fs_hz: int = 5000,
    segment_ms: int = 2000,
    epochs: int = 50,
    batch_size: int = 32,
    seed: int = 1337,
    device_preferred: str = "auto",
) -> Dict[str, Any]:
    print("[openalterego] CTC phoneme training")
    return {
        "task": "ctc_phoneme",
        **train_gowda_ctc(
            data_dir,
            fs_hz=int(fs_hz),
            segment_ms=int(segment_ms),
            epochs=int(epochs),
            batch_size=int(batch_size),
            seed=int(seed),
            device_preferred=device_preferred,
            save_path=data_dir / "ablations" / "ctc_gowda.pt",
        ),
    }


def run_phase2_all(
    data_dir: Path,
    *,
    fs_hz: int = 5000,
    device_preferred: str = "auto",
    seeds: List[int] | None = None,
) -> Dict[str, Any]:
    seeds = list(seeds or [1337, 1338, 1339, 1340, 1341])
    report: Dict[str, Any] = {
        "phase": 2,
        "session": str(data_dir),
        "sections": {},
    }
    report["sections"]["weekday_ceiling"] = run_weekday_ceiling(
        data_dir, fs_hz=int(fs_hz), device_preferred=device_preferred
    )
    report["sections"]["multiseed_bootstrap"] = run_multiseed_bootstrap(
        data_dir, fs_hz=int(fs_hz), seeds=seeds, device_preferred=device_preferred
    )
    report["sections"]["ctc_phoneme"] = run_ctc_path(
        data_dir, fs_hz=int(fs_hz), device_preferred=device_preferred
    )
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Gowda Phase 2 experiments")
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--fs", type=int, default=5000)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--out", type=str, default="")
    ap.add_argument(
        "--only",
        type=str,
        default="all",
        choices=["all", "weekday", "multiseed", "ctc"],
    )
    ap.add_argument("--seeds", type=str, default="1337,1338,1339,1340,1341")
    args = ap.parse_args()

    data_dir = Path(args.data)
    seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]

    if args.only == "weekday":
        report = {"sections": {"weekday_ceiling": run_weekday_ceiling(data_dir, fs_hz=int(args.fs), device_preferred=str(args.device))}}
    elif args.only == "multiseed":
        report = {
            "sections": {
                "multiseed_bootstrap": run_multiseed_bootstrap(
                    data_dir, fs_hz=int(args.fs), seeds=seeds, device_preferred=str(args.device)
                )
            }
        }
    elif args.only == "ctc":
        report = {"sections": {"ctc_phoneme": run_ctc_path(data_dir, fs_hz=int(args.fs), device_preferred=str(args.device))}}
    else:
        report = run_phase2_all(data_dir, fs_hz=int(args.fs), device_preferred=str(args.device), seeds=seeds)

    out = Path(args.out) if args.out else data_dir / "ablations" / "phase2_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
