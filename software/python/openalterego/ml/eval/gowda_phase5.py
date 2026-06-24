"""Phase 5: efficient SPD (diag+delta) + augment + lexicon Viterbi decode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from ..ctc.train import train_gowda_ctc
from .gowda_phase4 import run_decode_ablation


def run_spd_v3_efficient(
    data_dir: Path,
    *,
    fs_hz: int = 5000,
    epochs: int = 80,
    batch_size: int = 64,
    seed: int = 1337,
    device_preferred: str = "auto",
    feature_mode: str = "diag_delta",
) -> Dict[str, Any]:
    """Compact σ features + augment + 2-layer GRU for speed and accuracy."""
    print(f"[openalterego] SPD v3 efficient seed={seed} mode={feature_mode}")
    row = train_gowda_ctc(
        data_dir,
        fs_hz=int(fs_hz),
        feature_type="spd",
        feature_mode=str(feature_mode),
        augment_train=True,
        hidden=192,
        num_layers=2,
        dropout=0.15,
        epochs=int(epochs),
        batch_size=int(batch_size),
        lr=1.5e-3,
        warmup_epochs=5,
        seed=int(seed),
        device_preferred=device_preferred,
        val_decode_mode="greedy",
        eval_test=True,
        save_path=data_dir / "ablations" / f"ctc_spd_v3_{feature_mode}_seed{seed}.pt",
    )
    ckpt = Path(row["checkpoint"])
    decode = run_decode_ablation(data_dir, ckpt, device_preferred=device_preferred)
    return {"task": "spd_v3_efficient", "feature_mode": feature_mode, "decode_ablation": decode, **row}


def main() -> None:
    ap = argparse.ArgumentParser(description="Gowda Phase 5: efficient SPD v3")
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--fs", type=int, default=5000)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--feature-mode", type=str, default="diag_delta", choices=["diag", "diag_delta", "upper_tri"])
    args = ap.parse_args()

    data_dir = Path(args.data)
    report = {
        "phase": 5,
        "sections": {
            "spd_v3": run_spd_v3_efficient(
                data_dir,
                fs_hz=int(args.fs),
                epochs=int(args.epochs),
                seed=int(args.seed),
                device_preferred=str(args.device),
                feature_mode=str(args.feature_mode),
            )
        },
    }
    out = Path(args.out) if args.out else data_dir / "ablations" / "phase5_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[openalterego] phase5 report -> {out}")


if __name__ == "__main__":
    main()
