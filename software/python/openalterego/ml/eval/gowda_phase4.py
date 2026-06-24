"""Phase 4: beyond paper — beam decode, test split, enhanced SPD, multi-seed CI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..ctc.eval import eval_checkpoint
from ..ctc.train import train_gowda_ctc
from .bootstrap import summarize_ctc_multiseed


def run_decode_ablation(
    data_dir: Path,
    checkpoint: Path,
    *,
    device_preferred: str = "auto",
    beam_width: int = 50,
) -> Dict[str, Any]:
    """Compare greedy / beam / lexicon-beam on val and test for an existing checkpoint."""
    print(f"[openalterego] decode ablation: {checkpoint}")
    sections: Dict[str, Any] = {}
    for split in ("val", "test"):
        sections[split] = {}
        for mode in ("greedy", "beam", "beam_lexicon", "lexicon_viterbi"):
            row = eval_checkpoint(
                checkpoint,
                data_dir,
                split=split,  # type: ignore[arg-type]
                decode_mode=mode,  # type: ignore[arg-type]
                beam_width=int(beam_width),
                device_preferred=device_preferred,
            )
            sections[split][mode] = {k: v for k, v in row.items() if k not in ("hyp_words", "ref_words")}
    return {"task": "decode_ablation", "checkpoint": str(checkpoint), "sections": sections}


def run_spd_v2_train(
    data_dir: Path,
    *,
    fs_hz: int = 5000,
    epochs: int = 60,
    batch_size: int = 32,
    seed: int = 1337,
    device_preferred: str = "auto",
    use_upper_tri: bool = True,
) -> Dict[str, Any]:
    """Train enhanced SPD model; greedy val for checkpointing, beam ablation after."""
    print(f"[openalterego] SPD v2 train seed={seed} upper_tri={use_upper_tri}")
    row = train_gowda_ctc(
        data_dir,
        fs_hz=int(fs_hz),
        feature_type="spd",
        use_upper_tri=bool(use_upper_tri),
        hidden=256,
        num_layers=3,
        dropout=0.2,
        epochs=int(epochs),
        batch_size=int(batch_size),
        seed=int(seed),
        device_preferred=device_preferred,
        val_decode_mode="greedy",
        beam_width=50,
        eval_test=True,
        save_path=data_dir / "ablations" / f"ctc_spd_v2_seed{seed}.pt",
    )
    ckpt = data_dir / "ablations" / f"ctc_spd_v2_seed{seed}.pt"
    decode = run_decode_ablation(data_dir, ckpt, device_preferred=device_preferred)
    return {"task": "spd_v2_train", "decode_ablation": decode, **row}


def run_multiseed_spd(
    data_dir: Path,
    *,
    seeds: List[int],
    fs_hz: int = 5000,
    epochs: int = 60,
    device_preferred: str = "auto",
    use_upper_tri: bool = True,
    n_bootstrap: int = 2000,
) -> Dict[str, Any]:
    runs = []
    for seed in seeds:
        runs.append(
            run_spd_v2_train(
                data_dir,
                fs_hz=int(fs_hz),
                epochs=int(epochs),
                seed=int(seed),
                device_preferred=device_preferred,
                use_upper_tri=use_upper_tri,
            )
        )
    summary = summarize_ctc_multiseed(runs, split_key="test_beam", n_bootstrap=int(n_bootstrap))
    return {"task": "multiseed_spd_v2", **summary}


def run_phase4_all(
    data_dir: Path,
    *,
    fs_hz: int = 5000,
    device_preferred: str = "auto",
    seeds: Optional[List[int]] = None,
    legacy_checkpoint: Optional[Path] = None,
    skip_multiseed: bool = False,
    epochs: int = 60,
) -> Dict[str, Any]:
    data_dir = Path(data_dir)
    seeds = list(seeds or [1337, 1338, 1339])
    report: Dict[str, Any] = {"phase": 4, "session": str(data_dir), "sections": {}}

    legacy = legacy_checkpoint or (data_dir / "ablations" / "ctc_spd.pt")
    if legacy.is_file():
        report["sections"]["decode_ablation_legacy"] = run_decode_ablation(
            data_dir, legacy, device_preferred=device_preferred
        )

    if not skip_multiseed:
        report["sections"]["multiseed_spd_v2"] = run_multiseed_spd(
            data_dir,
            seeds=seeds,
            fs_hz=int(fs_hz),
            epochs=int(epochs),
            device_preferred=device_preferred,
        )
    else:
        report["sections"]["spd_v2_single"] = run_spd_v2_train(
            data_dir,
            fs_hz=int(fs_hz),
            epochs=int(epochs),
            device_preferred=device_preferred,
        )

    best_ckpt = data_dir / "ablations" / f"ctc_spd_v2_seed{seeds[0]}.pt"
    if best_ckpt.is_file():
        report["sections"]["decode_ablation_v2"] = run_decode_ablation(
            data_dir, best_ckpt, device_preferred=device_preferred
        )
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Gowda Phase 4: beyond paper")
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--fs", type=int, default=5000)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--seeds", type=str, default="1337,1338,1339")
    ap.add_argument("--checkpoint", type=str, default="")
    ap.add_argument("--beam-width", type=int, default=50)
    ap.add_argument(
        "--only",
        type=str,
        default="all",
        choices=["all", "decode", "train", "multiseed"],
    )
    ap.add_argument("--skip-multiseed", action="store_true")
    ap.add_argument("--no-upper-tri", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data)
    seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]
    ckpt = Path(args.checkpoint) if str(args.checkpoint).strip() else data_dir / "ablations" / "ctc_spd.pt"

    if args.only == "decode":
        report = {"sections": {"decode_ablation": run_decode_ablation(data_dir, ckpt, device_preferred=str(args.device), beam_width=int(args.beam_width))}}
    elif args.only == "train":
        report = {
            "sections": {
                "spd_v2_single": run_spd_v2_train(
                    data_dir,
                    fs_hz=int(args.fs),
                    epochs=int(args.epochs),
                    seed=int(args.seed),
                    device_preferred=str(args.device),
                    use_upper_tri=not bool(args.no_upper_tri),
                )
            }
        }
    elif args.only == "multiseed":
        report = {
            "sections": {
                "multiseed_spd_v2": run_multiseed_spd(
                    data_dir,
                    seeds=seeds,
                    fs_hz=int(args.fs),
                    epochs=int(args.epochs),
                    device_preferred=str(args.device),
                    use_upper_tri=not bool(args.no_upper_tri),
                )
            }
        }
    else:
        report = run_phase4_all(
            data_dir,
            fs_hz=int(args.fs),
            device_preferred=str(args.device),
            seeds=seeds,
            legacy_checkpoint=ckpt if ckpt.is_file() else None,
            skip_multiseed=bool(args.skip_multiseed),
            epochs=int(args.epochs),
        )

    out = Path(args.out) if args.out else data_dir / "ablations" / "phase4_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[openalterego] phase4 report -> {out}")


if __name__ == "__main__":
    main()
