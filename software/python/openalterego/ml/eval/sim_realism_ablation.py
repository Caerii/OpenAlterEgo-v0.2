"""Realism preset + SNR calibration ablations for sim→real transfer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ...sim.dataset import generate_dataset
from ...sim.metrics.realism_match import (
    RealismVariant,
    default_realism_variants,
    match_score,
    parse_variant_tags,
    real_gowda_baseline_stats,
    sim_gowda_variant_stats,
)
from ...sim.scenarios.gowda_small_vocab import build_gowda_dataset_config
from .sim_transfer import eval_sim_only_anchor


def run_probe_ablation(
    real_dir: Path,
    variants: Sequence[RealismVariant],
    *,
    probe_trials: int = 8,
    seed: int = 1337,
) -> Dict[str, Any]:
    """Fast signal-statistics ladder vs real OSF train events."""
    real_dir = Path(real_dir)
    real_stats = real_gowda_baseline_stats(real_dir, seed=seed)
    rows: list[dict[str, Any]] = []
    for variant in variants:
        sim_stats, extra = sim_gowda_variant_stats(
            variant, probe_trials=int(probe_trials), seed=int(seed)
        )
        scores = match_score(sim_stats, real_stats)
        rows.append(
            {
                "variant": variant.to_dict(),
                "sim_stats": sim_stats.to_dict(),
                "match": scores,
                "probe_meta": {
                    "snr_calibration": extra["meta"].get("snr_calibration"),
                    "noise_scale": extra["meta"].get("noise_scale"),
                    "realism_preset": extra["meta"].get("realism_preset"),
                },
            }
        )
    rows.sort(key=lambda r: float(r["match"]["total"]))
    return {
        "real_dir": str(real_dir),
        "real_baseline": real_stats.to_dict(),
        "probe_trials": int(probe_trials),
        "variants": rows,
    }


def _ensure_sim_corpus(
    variant: RealismVariant,
    out_dir: Path,
    *,
    n_trials: int,
    seed: int,
    force: bool = False,
) -> Path:
    out_dir = Path(out_dir)
    if not force and (out_dir / "events.csv").is_file() and (out_dir / "signals.npy").is_file():
        return out_dir
    ds = build_gowda_dataset_config(
        out_dir,
        n_trials=int(n_trials),
        seed=int(seed),
        realism=str(variant.preset),
        snr_target_db=variant.snr_target_db,
        snr_motion_target_db=variant.snr_motion_target_db,
    )
    return generate_dataset(ds)


def run_transfer_ablation(
    real_dir: Path,
    variants: Sequence[RealismVariant],
    *,
    corpus_root: Path,
    n_trials: int = 100,
    pretrain_epochs: int = 30,
    anchor_epochs: int = 15,
    seed: int = 1337,
    device_preferred: str = "auto",
    feature_mode: str = "diag_delta",
    decode_mode: str = "trial_lm",
    force_regen: bool = False,
) -> Dict[str, Any]:
    """Generate sim corpora per variant and run sim-only + anchor transfer."""
    real_dir = Path(real_dir)
    corpus_root = Path(corpus_root)
    corpus_root.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    for variant in variants:
        sim_dir = corpus_root / variant.tag
        print(f"[openalterego] realism transfer variant={variant.tag} sim_dir={sim_dir}")
        _ensure_sim_corpus(
            variant, sim_dir, n_trials=int(n_trials), seed=int(seed), force=bool(force_regen)
        )
        transfer = eval_sim_only_anchor(
            sim_dir,
            real_dir,
            pretrain_epochs=int(pretrain_epochs),
            anchor_epochs=int(anchor_epochs),
            seed=int(seed),
            device_preferred=device_preferred,
            feature_mode=str(feature_mode),
            decode_mode=str(decode_mode),
        )
        sim_only = next((r for r in transfer["runs"] if r["tag"] == "sim_only"), None)
        anchor = next((r for r in transfer["runs"] if r["tag"] == "sim_pretrain_real_anchor"), None)
        runs.append(
            {
                "variant": variant.to_dict(),
                "sim_dir": str(sim_dir),
                "sim_only": sim_only,
                "anchor": anchor,
            }
        )
    runs.sort(
        key=lambda r: float(
            (r.get("anchor") or r.get("sim_only") or {}).get("real_test", {}).get("wer", 1.0)
        )
    )
    return {
        "real_dir": str(real_dir),
        "corpus_root": str(corpus_root),
        "n_trials": int(n_trials),
        "runs": runs,
    }


def run_realism_ablation(
    real_dir: Path,
    *,
    out_dir: Optional[Path] = None,
    variants: Optional[Sequence[RealismVariant]] = None,
    probe_trials: int = 8,
    transfer_trials: int = 100,
    run_transfer: bool = False,
    top_k_transfer: Optional[int] = None,
    pretrain_epochs: int = 30,
    anchor_epochs: int = 15,
    seed: int = 1337,
    device_preferred: str = "auto",
    feature_mode: str = "diag_delta",
    decode_mode: str = "trial_lm",
    force_regen: bool = False,
) -> Dict[str, Any]:
    """Probe all variants; optionally transfer on all or top-k probe matches."""
    real_dir = Path(real_dir)
    out_dir = Path(out_dir) if out_dir is not None else real_dir / "ablations" / "realism_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    variant_list = list(variants if variants is not None else default_realism_variants())

    report: Dict[str, Any] = {
        "real_dir": str(real_dir),
        "out_dir": str(out_dir),
        "seed": int(seed),
    }
    probe = run_probe_ablation(
        real_dir, variant_list, probe_trials=int(probe_trials), seed=int(seed)
    )
    report["probe"] = probe

    if run_transfer:
        transfer_variants = list(variant_list)
        if top_k_transfer is not None and int(top_k_transfer) > 0:
            ranked = [RealismVariant(**row["variant"]) for row in probe["variants"]]
            transfer_variants = ranked[: int(top_k_transfer)]
        transfer = run_transfer_ablation(
            real_dir,
            transfer_variants,
            corpus_root=out_dir / "corpus",
            n_trials=int(transfer_trials),
            pretrain_epochs=int(pretrain_epochs),
            anchor_epochs=int(anchor_epochs),
            seed=int(seed),
            device_preferred=device_preferred,
            feature_mode=str(feature_mode),
            decode_mode=str(decode_mode),
            force_regen=bool(force_regen),
        )
        report["transfer"] = transfer

    report_path = out_dir / "realism_ablation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[openalterego] realism ablation report -> {report_path}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Realism preset + SNR ablation harness")
    ap.add_argument("--real", type=str, required=True, help="Real Gowda OSF session")
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--probe-trials", type=int, default=8)
    ap.add_argument("--probe-only", action="store_true", help="Skip sim-transfer training")
    ap.add_argument("--transfer", action="store_true", help="Run sim-only + anchor per variant")
    ap.add_argument("--top-k", type=int, default=0, help="Transfer only top-k probe matches (0=all)")
    ap.add_argument("--trials", type=int, default=100, help="Sim corpus trials per transfer variant")
    ap.add_argument("--pretrain-epochs", type=int, default=30)
    ap.add_argument("--anchor-epochs", type=int, default=15)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--feature-mode", type=str, default="diag_delta")
    ap.add_argument("--decode-mode", type=str, default="trial_lm")
    ap.add_argument("--variants", type=str, default="", help="Comma-separated variant tags")
    ap.add_argument("--force-regen", action="store_true")
    args = ap.parse_args()

    tags = [x.strip() for x in str(args.variants).split(",") if x.strip()]
    variants = parse_variant_tags(tags if tags else None)
    out = Path(args.out) if args.out else Path(args.real) / "ablations" / "realism_ablation"
    run_realism_ablation(
        Path(args.real),
        out_dir=out,
        variants=variants,
        probe_trials=int(args.probe_trials),
        transfer_trials=int(args.trials),
        run_transfer=bool(args.transfer) and not bool(args.probe_only),
        top_k_transfer=int(args.top_k) if int(args.top_k) > 0 else None,
        pretrain_epochs=int(args.pretrain_epochs),
        anchor_epochs=int(args.anchor_epochs),
        seed=int(args.seed),
        device_preferred=str(args.device),
        feature_mode=str(args.feature_mode),
        decode_mode=str(args.decode_mode),
        force_regen=bool(args.force_regen),
    )


if __name__ == "__main__":
    main()
