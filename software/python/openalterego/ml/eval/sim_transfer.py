"""Sim→real transfer evaluation: train on biophysical Gowda-shaped corpus, test on OSF EMG."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from ..ctc.eval import build_ctc_model_from_checkpoint
from ..ctc.train import train_gowda_ctc
from ..ctc.trial_decode import evaluate_ctc_trial_lm, tune_trial_lm_weight
from ..ctc.trial_lm import fit_trial_lm
from ..data_split import gowda_official_train_val_test_indices
from ..datasets.events import load_gowda_events, sanitize_trial_events
from ..device import resolve_device
from .gowda_phase6 import _build_dataset


def _trial_fraction_events(events: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    """Take a fraction of training trials (all 4 words per trial)."""
    frac = float(np.clip(frac, 0.0, 1.0))
    if frac <= 0.0:
        return events.iloc[0:0].reset_index(drop=True)
    if frac >= 1.0:
        return events.reset_index(drop=True)
    trial_ids = events["trial_id"].astype(int).values
    tr_trials = sorted(set(trial_ids[trial_ids < 370]))
    rng = random.Random(int(seed))
    rng.shuffle(tr_trials)
    n_take = max(1, int(round(len(tr_trials) * frac)))
    keep = set(tr_trials[:n_take])
    mask = events["trial_id"].astype(int).isin(keep)
    return events.loc[mask].reset_index(drop=True)


def _merge_sim_real_train(sim_dir: Path, real_dir: Path, real_frac: float, seed: int) -> Path:
    """Write combined train session under ``real_dir/ablations/sim_transfer_merged``."""
    sim_ev = sanitize_trial_events(pd.read_csv(sim_dir / "events.csv"))
    real_ev = load_gowda_events(real_dir)
    real_sub = _trial_fraction_events(real_ev, real_frac, seed)

    sim_sig = np.load(sim_dir / "signals.npy", mmap_mode="r")
    real_sig = np.load(real_dir / "signals.npy", mmap_mode="r")

    # Concatenate signals; offset real event samples.
    offset = int(sim_sig.shape[0])
    sig = np.concatenate([np.asarray(sim_sig, dtype=np.float32), np.asarray(real_sig, dtype=np.float32)], axis=0)

    sim_rows = sim_ev.copy()
    if "trial_id" not in sim_rows.columns or sim_rows.empty:
        raise ValueError("sim events.csv missing trial_id; use --scenario gowda_sv")
    sim_trials = int(sim_rows["trial_id"].max()) + 1

    real_rows = real_sub.copy()
    real_rows["start_sample"] = real_rows["start_sample"].astype(int) + offset
    real_rows["end_sample"] = real_rows["end_sample"].astype(int) + offset
    real_rows["trial_id"] = real_rows["trial_id"].astype(int) + sim_trials

    merged = pd.concat([sim_rows, real_rows], ignore_index=True)
    out = real_dir / "ablations" / f"sim_transfer_merged_frac{real_frac:.2f}_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "signals.npy", sig)
    merged.to_csv(out / "events.csv", index=False)
    meta = {
        "fs_hz": int(json.loads((sim_dir / "meta.json").read_text())["fs_hz"]),
        "channels": int(sig.shape[1]),
        "sim_dir": str(sim_dir),
        "real_frac": float(real_frac),
        "n_sim_events": int(len(sim_rows)),
        "n_real_events": int(len(real_rows)),
        "sim_trials": int(sim_trials),
        "split_mode": "sim_transfer_merged",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


def eval_checkpoint_on_real(
    checkpoint: Path,
    real_dir: Path,
    *,
    device_preferred: str = "auto",
    decode_mode: str = "trial_lm",
) -> Dict[str, Any]:
    device = resolve_device(device_preferred)
    ckpt = torch.load(Path(checkpoint), map_location=device, weights_only=False)
    model = build_ctc_model_from_checkpoint(ckpt, device)
    events = pd.read_csv(real_dir / "events.csv")
    lm = fit_trial_lm(events)

    if decode_mode == "trial_lm":
        te_idx = gowda_official_train_val_test_indices(events["trial_id"].astype(int).values)[2]
        te_events = events.iloc[te_idx].reset_index(drop=True)
        ds = _build_dataset(real_dir, te_events, ckpt)
        lm_w = tune_trial_lm_weight(model, _build_dataset(real_dir, events.iloc[gowda_official_train_val_test_indices(events["trial_id"].astype(int).values)[1]].reset_index(drop=True), ckpt), device, lm)
        return evaluate_ctc_trial_lm(model, ds, device, lm, lm_weight=float(lm_w))

    from ..ctc.eval import eval_checkpoint

    return eval_checkpoint(
        checkpoint,
        real_dir,
        split="test",
        decode_mode=decode_mode,  # type: ignore[arg-type]
        device_preferred=device_preferred,
    )


def run_sim_transfer(
    sim_dir: Path,
    real_dir: Path,
    *,
    real_fracs: Optional[List[float]] = None,
    pretrain_epochs: int = 30,
    finetune_epochs: int = 15,
    anchor_epochs: int = 15,
    anchor_after_sim: bool = True,
    seed: int = 1337,
    device_preferred: str = "auto",
    feature_mode: str = "diag_delta",
    decode_mode: str = "trial_lm",
) -> Dict[str, Any]:
    """Train sim-only and sim+real fractions; evaluate on real test split."""
    sim_dir = Path(sim_dir)
    real_dir = Path(real_dir)
    real_fracs = list(real_fracs if real_fracs is not None else [0.0, 0.1, 0.5, 1.0])

    print(f"[openalterego] sim-transfer sim={sim_dir} real={real_dir} (SPD basis from real)")
    results: Dict[str, Any] = {
        "sim_dir": str(sim_dir),
        "real_dir": str(real_dir),
        "basis_session_dir": str(real_dir),
        "runs": [],
    }

    train_kw = dict(
        fs_hz=5000,
        feature_type="spd",
        feature_mode=str(feature_mode),
        hidden=256,
        num_layers=3,
        dropout=0.2,
        batch_size=32,
        warmup_epochs=3,
        seed=int(seed),
        device_preferred=device_preferred,
        val_decode_mode="greedy",
        eval_test=False,
        basis_session_dir=real_dir,
    )

    for frac in real_fracs:
        if float(frac) <= 0.0:
            train_dir = sim_dir
            tag = "sim_only"
            epochs = int(pretrain_epochs)
            lr = 1.5e-3
            augment = False
        else:
            train_dir = _merge_sim_real_train(sim_dir, real_dir, float(frac), seed)
            tag = f"sim_plus_real_{frac:.2f}"
            epochs = int(finetune_epochs)
            lr = 8e-4
            augment = True

        ckpt_path = real_dir / "ablations" / f"sim_transfer_{tag}_seed{seed}.pt"
        print(f"[openalterego] training {tag} -> {ckpt_path}")
        train_row = train_gowda_ctc(
            train_dir,
            augment_train=augment,
            epochs=epochs,
            lr=lr,
            save_path=ckpt_path,
            **train_kw,
        )

        if decode_mode == "trial_lm":
            metrics = eval_checkpoint_on_real(
                ckpt_path, real_dir, device_preferred=device_preferred, decode_mode="trial_lm"
            )
        else:
            metrics = eval_checkpoint_on_real(
                ckpt_path, real_dir, device_preferred=device_preferred, decode_mode=str(decode_mode)
            )

        run = {
            "tag": tag,
            "real_frac": float(frac),
            "train_dir": str(train_dir),
            "checkpoint": str(ckpt_path),
            "train": {k: v for k, v in train_row.items() if k != "state_dict"},
            "real_test": {k: v for k, v in metrics.items() if k not in ("hyp_words", "ref_words")},
        }
        results["runs"].append(run)
        print(
            f"[openalterego] {tag}: real test WER={run['real_test'].get('wer')} "
            f"acc={run['real_test'].get('word_acc')}"
        )

        if float(frac) <= 0.0 and bool(anchor_after_sim):
            anchor_ckpt = real_dir / "ablations" / f"sim_transfer_sim_pretrain_anchor_seed{seed}.pt"
            print(f"[openalterego] anchor finetune on real -> {anchor_ckpt}")
            anchor_train = train_gowda_ctc(
                real_dir,
                augment_train=True,
                epochs=int(anchor_epochs),
                lr=8e-4,
                save_path=anchor_ckpt,
                init_checkpoint=ckpt_path,
                **train_kw,
            )
            anchor_metrics = eval_checkpoint_on_real(
                anchor_ckpt, real_dir, device_preferred=device_preferred, decode_mode=str(decode_mode)
            )
            results["runs"].append(
                {
                    "tag": "sim_pretrain_real_anchor",
                    "real_frac": 0.0,
                    "train_dir": str(real_dir),
                    "checkpoint": str(anchor_ckpt),
                    "init_checkpoint": str(ckpt_path),
                    "train": {k: v for k, v in anchor_train.items() if k != "state_dict"},
                    "real_test": {
                        k: v for k, v in anchor_metrics.items() if k not in ("hyp_words", "ref_words")
                    },
                }
            )
            print(
                f"[openalterego] sim_pretrain_real_anchor: real test WER="
                f"{anchor_metrics.get('wer')} acc={anchor_metrics.get('word_acc')}"
            )

    return results


def eval_sim_only_anchor(
    sim_dir: Path,
    real_dir: Path,
    *,
    pretrain_epochs: int = 30,
    anchor_epochs: int = 15,
    seed: int = 1337,
    device_preferred: str = "auto",
    feature_mode: str = "diag_delta",
    decode_mode: str = "trial_lm",
) -> Dict[str, Any]:
    """Train sim-only then anchor finetune on real; evaluate both on real test."""
    report = run_sim_transfer(
        Path(sim_dir),
        Path(real_dir),
        real_fracs=[0.0],
        pretrain_epochs=int(pretrain_epochs),
        finetune_epochs=0,
        anchor_epochs=int(anchor_epochs),
        anchor_after_sim=True,
        seed=int(seed),
        device_preferred=device_preferred,
        feature_mode=str(feature_mode),
        decode_mode=str(decode_mode),
    )
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Sim→real CTC transfer harness")
    ap.add_argument("--sim", type=str, required=True, help="Gowda-shaped sim session directory")
    ap.add_argument("--real", type=str, required=True, help="Real Gowda OSF session (e.g. gowda_sv_full)")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--pretrain-epochs", type=int, default=30)
    ap.add_argument("--finetune-epochs", type=int, default=15)
    ap.add_argument("--feature-mode", type=str, default="diag_delta")
    ap.add_argument("--real-fracs", type=str, default="0,0.1,0.5,1.0")
    ap.add_argument("--decode-mode", type=str, default="trial_lm")
    ap.add_argument("--anchor-epochs", type=int, default=15)
    ap.add_argument("--no-anchor", action="store_true", help="Skip sim-pretrain → real anchor finetune")
    args = ap.parse_args()

    fracs = [float(x.strip()) for x in str(args.real_fracs).split(",") if x.strip()]
    report = run_sim_transfer(
        Path(args.sim),
        Path(args.real),
        real_fracs=fracs,
        pretrain_epochs=int(args.pretrain_epochs),
        finetune_epochs=int(args.finetune_epochs),
        anchor_epochs=int(args.anchor_epochs),
        anchor_after_sim=not bool(args.no_anchor),
        seed=int(args.seed),
        device_preferred=str(args.device),
        feature_mode=str(args.feature_mode),
        decode_mode=str(args.decode_mode),
    )
    out = Path(args.out) if args.out else Path(args.real) / "ablations" / "sim_transfer_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[openalterego] sim-transfer report -> {out}")


if __name__ == "__main__":
    main()
