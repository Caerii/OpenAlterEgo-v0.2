"""Phase 3: full small-vocab import + SPD σ(τ) CTC (paper feature path)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..ctc.train import train_gowda_ctc
from ..datasets.gowda import import_gowda_small_vocab_from_osf


def import_full_small_vocab(
    out_dir: Path,
    *,
    download_dir: Optional[Path] = None,
    fs_hz: int = 5000,
) -> Dict[str, Any]:
    """Import all OSF small-vocab labels (no top-N filter)."""
    print(f"[openalterego] full small-vocab import -> {out_dir}")
    rep = import_gowda_small_vocab_from_osf(
        Path(out_dir),
        download_dir=download_dir,
        fs_hz=float(fs_hz),
        top_labels=None,
        min_samples_per_label=1,
    )
    return {"task": "import_full_vocab", **rep.to_dict()}


def run_spd_ctc(
    data_dir: Path,
    *,
    fs_hz: int = 5000,
    segment_ms: int = 2000,
    epochs: int = 50,
    batch_size: int = 32,
    seed: int = 1337,
    device_preferred: str = "auto",
) -> Dict[str, Any]:
    print("[openalterego] SPD sigma(tau) + GRU + CTC training")
    return {
        "task": "spd_ctc_phoneme",
        **train_gowda_ctc(
            data_dir,
            fs_hz=int(fs_hz),
            segment_ms=int(segment_ms),
            feature_type="spd",
            epochs=int(epochs),
            batch_size=int(batch_size),
            seed=int(seed),
            device_preferred=device_preferred,
            save_path=data_dir / "ablations" / "ctc_spd.pt",
        ),
    }


def run_raw_ctc_baseline(
    data_dir: Path,
    *,
    fs_hz: int = 5000,
    segment_ms: int = 2000,
    epochs: int = 50,
    batch_size: int = 32,
    seed: int = 1337,
    device_preferred: str = "auto",
) -> Dict[str, Any]:
    print("[openalterego] raw EMG CNN+CTC baseline (comparison)")
    return {
        "task": "raw_ctc_baseline",
        **train_gowda_ctc(
            data_dir,
            fs_hz=int(fs_hz),
            segment_ms=int(segment_ms),
            feature_type="raw",
            epochs=int(epochs),
            batch_size=int(batch_size),
            seed=int(seed),
            device_preferred=device_preferred,
            save_path=data_dir / "ablations" / "ctc_raw_full.pt",
        ),
    }


def run_phase3_all(
    data_dir: Path,
    *,
    fs_hz: int = 5000,
    device_preferred: str = "auto",
    import_if_missing: bool = True,
    download_dir: Optional[Path] = None,
    skip_import: bool = False,
    skip_raw_baseline: bool = False,
    epochs: int = 50,
    seed: int = 1337,
) -> Dict[str, Any]:
    data_dir = Path(data_dir)
    report: Dict[str, Any] = {
        "phase": 3,
        "session": str(data_dir),
        "sections": {},
    }

    if not skip_import and import_if_missing and not (data_dir / "events.csv").is_file():
        report["sections"]["import"] = import_full_small_vocab(
            data_dir,
            download_dir=download_dir,
            fs_hz=int(fs_hz),
        )

    report["sections"]["spd_ctc"] = run_spd_ctc(
        data_dir,
        fs_hz=int(fs_hz),
        epochs=int(epochs),
        seed=int(seed),
        device_preferred=device_preferred,
    )
    if not skip_raw_baseline:
        report["sections"]["raw_ctc_baseline"] = run_raw_ctc_baseline(
            data_dir,
            fs_hz=int(fs_hz),
            epochs=int(epochs),
            seed=int(seed),
            device_preferred=device_preferred,
        )
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Gowda Phase 3: full vocab + SPD CTC")
    ap.add_argument("--data", type=str, required=True, help="Session dir (e.g. sessions/gowda_sv_full)")
    ap.add_argument("--fs", type=int, default=5000)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--import-only", action="store_true")
    ap.add_argument("--skip-import", action="store_true")
    ap.add_argument("--skip-raw-baseline", action="store_true")
    ap.add_argument("--download-dir", type=str, default="")
    ap.add_argument(
        "--only",
        type=str,
        default="all",
        choices=["all", "import", "spd", "raw_ctc"],
    )
    args = ap.parse_args()

    data_dir = Path(args.data)
    dl_dir = Path(args.download_dir) if str(args.download_dir).strip() else None

    if args.only == "import" or args.import_only:
        report = {"sections": {"import": import_full_small_vocab(data_dir, download_dir=dl_dir, fs_hz=int(args.fs))}}
    elif args.only == "spd":
        report = {"sections": {"spd_ctc": run_spd_ctc(data_dir, fs_hz=int(args.fs), device_preferred=str(args.device), epochs=int(args.epochs), seed=int(args.seed))}}
    elif args.only == "raw_ctc":
        report = {"sections": {"raw_ctc_baseline": run_raw_ctc_baseline(data_dir, fs_hz=int(args.fs), device_preferred=str(args.device), epochs=int(args.epochs), seed=int(args.seed))}}
    else:
        report = run_phase3_all(
            data_dir,
            fs_hz=int(args.fs),
            device_preferred=str(args.device),
            skip_import=bool(args.skip_import),
            skip_raw_baseline=bool(args.skip_raw_baseline),
            download_dir=dl_dir,
            epochs=int(args.epochs),
            seed=int(args.seed),
        )

    out = Path(args.out) if args.out else data_dir / "ablations" / "phase3_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[openalterego] phase3 report -> {out}")


if __name__ == "__main__":
    main()
