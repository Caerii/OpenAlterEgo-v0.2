"""Phase 6: deep error analysis + trial-context LM decode."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from ..ctc.eval import _events_for_split, build_ctc_model_from_checkpoint, eval_checkpoint
from ..ctc.dataset import PhonemeCTCDataset
from ..ctc.trial_decode import evaluate_ctc_trial_lm, tune_trial_lm_weight
from ..ctc.trial_lm import WEEKDAYS, MONTHS, fit_trial_lm
from ..device import resolve_device
from ..spd.basis import ensure_gowda_spd_basis


YEAR_PREFIXES = ("nineteen_", "two_thousand_")


def _error_category(ref: str, hyp: str, word_idx: int) -> str:
    if ref == hyp:
        return "correct"
    if int(word_idx) == 0 and ref in WEEKDAYS:
        return "weekday"
    if int(word_idx) == 1 and ref in MONTHS:
        return "month"
    if int(word_idx) == 2:
        return "ordinal"
    if int(word_idx) == 3 or ref.startswith(YEAR_PREFIXES) or "thousand" in ref:
        if hyp and ref.startswith(hyp):
            return "year_truncation"
        if hyp and hyp.startswith(ref):
            return "year_overgen"
        if ref.split("_")[0] == hyp.split("_")[0] if hyp else False:
            return "year_same_era"
        return "year_other"
    return "other"


def analyze_errors(
    hyp_words: List[str],
    ref_words: List[str],
    events: pd.DataFrame,
) -> Dict[str, Any]:
    assert len(hyp_words) == len(ref_words) == len(events)
    pairs: Counter[str] = Counter()
    by_slot: Dict[int, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "errors": 0})
    by_cat: Counter[str] = Counter()
    errors: List[Dict[str, Any]] = []

    for i, (h, r) in enumerate(zip(hyp_words, ref_words)):
        wi = int(events.iloc[i]["word_idx"])
        cat = _error_category(r, h, wi)
        by_cat[cat] += 1 if h != r else 0
        if h == r:
            by_slot[wi]["correct"] += 1
        else:
            by_slot[wi]["errors"] += 1
            key = f"{r} -> {h}"
            pairs[key] += 1
            errors.append(
                {
                    "trial_id": int(events.iloc[i]["trial_id"]),
                    "word_idx": wi,
                    "ref": r,
                    "hyp": h,
                    "category": cat,
                }
            )

    n = len(ref_words)
    n_err = sum(1 for h, r in zip(hyp_words, ref_words) if h != r)
    year_errs = [e for e in errors if e["category"].startswith("year")]
    return {
        "n": n,
        "n_errors": n_err,
        "wer": round(n_err / max(n, 1), 4),
        "by_category": dict(by_cat),
        "by_slot": {str(k): v for k, v in by_slot.items()},
        "top_confusions": pairs.most_common(20),
        "year_error_fraction": round(len(year_errs) / max(n_err, 1), 4),
        "sample_errors": errors[:40],
    }


def _build_dataset(data_dir: Path, events: pd.DataFrame, ckpt: Dict[str, Any]) -> PhonemeCTCDataset:
    signals = np.load(data_dir / "signals.npy", mmap_mode="r")
    ft = str(ckpt.get("feature_type", "raw"))
    spd_basis = None
    if ft == "spd":
        spd_basis = ensure_gowda_spd_basis(
            data_dir,
            fs_hz=int(ckpt.get("fs", 5000)),
            emg_mode=str(ckpt.get("emg_mode", "gowda")),
            seed=int(ckpt.get("seed", 1337)),
            use_upper_tri=bool(ckpt.get("use_upper_tri", False)),
            feature_mode=str(ckpt.get("feature_mode", "full")),
        )
    return PhonemeCTCDataset(
        np.asarray(signals, dtype=np.float32),
        events,
        fs_hz=int(ckpt.get("fs", 5000)),
        segment_ms=int(ckpt.get("segment_ms", 2000)),
        seed=int(ckpt.get("seed", 1337)),
        emg_mode=str(ckpt.get("emg_mode", "gowda")),
        per_event_preprocess=True,
        feature_type=ft,  # type: ignore[arg-type]
        spd_basis=spd_basis,
        session_dir=str(data_dir),
        use_upper_tri=bool(ckpt.get("use_upper_tri", False)),
        feature_mode=str(ckpt.get("feature_mode", "full")),
    )


def run_trial_lm_decode(
    data_dir: Path,
    checkpoint: Path,
    *,
    device_preferred: str = "auto",
    topk: int = 16,
) -> Dict[str, Any]:
    data_dir = Path(data_dir)
    device = resolve_device(device_preferred)
    ckpt = torch.load(Path(checkpoint), map_location=device, weights_only=False)
    model = build_ctc_model_from_checkpoint(ckpt, device)

    events = pd.read_csv(data_dir / "events.csv")
    lm = fit_trial_lm(events)

    val_events = _events_for_split(events, "val")
    val_ds = _build_dataset(data_dir, val_events, ckpt)
    lm_weight = tune_trial_lm_weight(model, val_ds, device, lm, topk=int(topk))
    print(f"[openalterego] trial LM weight tuned on val: {lm_weight}")

    sections: Dict[str, Any] = {"lm_weight": lm_weight, "topk": int(topk)}
    for split in ("val", "test"):
        split_events = _events_for_split(events, split)
        ds = _build_dataset(data_dir, split_events, ckpt)
        metrics = evaluate_ctc_trial_lm(
            model, ds, device, lm, lm_weight=float(lm_weight), topk=int(topk)
        )
        err = analyze_errors(metrics["hyp_words"], metrics["ref_words"], split_events)
        sections[split] = {
            k: v for k, v in metrics.items() if k not in ("hyp_words", "ref_words")
        }
        sections[split]["error_analysis"] = err
    return {"task": "trial_lm_decode", "checkpoint": str(checkpoint), "sections": sections}


def run_error_analysis_baseline(
    data_dir: Path,
    checkpoint: Path,
    *,
    decode_mode: str = "beam_lexicon",
    device_preferred: str = "auto",
) -> Dict[str, Any]:
    row = eval_checkpoint(
        checkpoint,
        data_dir,
        split="test",
        decode_mode=decode_mode,  # type: ignore[arg-type]
        device_preferred=device_preferred,
    )
    events = _events_for_split(pd.read_csv(Path(data_dir) / "events.csv"), "test")
    err = analyze_errors(row["hyp_words"], row["ref_words"], events)
    return {
        "task": "error_analysis",
        "decode_mode": decode_mode,
        "checkpoint": str(checkpoint),
        "metrics": {k: v for k, v in row.items() if k not in ("hyp_words", "ref_words")},
        "error_analysis": err,
    }


def run_phase6_all(
    data_dir: Path,
    checkpoint: Path,
    *,
    device_preferred: str = "auto",
) -> Dict[str, Any]:
    return {
        "phase": 6,
        "sections": {
            "baseline_errors": run_error_analysis_baseline(
                data_dir, checkpoint, device_preferred=device_preferred
            ),
            "trial_lm": run_trial_lm_decode(
                data_dir, checkpoint, device_preferred=device_preferred
            ),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Gowda Phase 6: error analysis + trial LM")
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, default="")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--only", type=str, default="all", choices=["all", "errors", "trial_lm"])
    args = ap.parse_args()

    data_dir = Path(args.data)
    ckpt = Path(args.checkpoint) if args.checkpoint else data_dir / "ablations" / "ctc_spd_v3_diag_delta_seed1337.pt"

    if args.only == "errors":
        report = {"sections": {"baseline_errors": run_error_analysis_baseline(data_dir, ckpt, device_preferred=str(args.device))}}
    elif args.only == "trial_lm":
        report = {"sections": {"trial_lm": run_trial_lm_decode(data_dir, ckpt, device_preferred=str(args.device))}}
    else:
        report = run_phase6_all(data_dir, ckpt, device_preferred=str(args.device))

    out = Path(args.out) if args.out else data_dir / "ablations" / "phase6_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[openalterego] phase6 report -> {out}")


if __name__ == "__main__":
    main()
