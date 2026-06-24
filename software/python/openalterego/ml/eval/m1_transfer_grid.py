"""M1 phoneme-synth sim→real transfer ablation grid."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ...sim.dataset import generate_dataset
from ...sim.scenarios.gowda_small_vocab import build_gowda_dataset_config
from .sim_transfer import eval_sim_only_anchor


@dataclass(frozen=True)
class M1GridVariant:
    tag: str
    trials: int = 100
    realism: str = "wearable"
    snr_target_db: Optional[float] = 18.9
    snr_motion_target_db: Optional[float] = 12.7
    phone_templates_path: Optional[str] = None
    coarticulation_enabled: bool = True
    seed: int = 1337

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_m1_grid() -> List[M1GridVariant]:
    tpl = "./sessions/gowda_sv_full/phone_templates.json"
    tpl_corpus = "./sessions/gowda_sv_full/phone_templates_corpus.json"
    return [
        M1GridVariant("m1_coart_500", trials=500, phone_templates_path=tpl),
        M1GridVariant("m1_coart_100", trials=100, phone_templates_path=tpl),
        M1GridVariant(
            "m1_nocoart_100",
            trials=100,
            phone_templates_path=tpl,
            coarticulation_enabled=False,
        ),
        M1GridVariant(
            "m1_off_100",
            trials=100,
            realism="off",
            snr_target_db=None,
            snr_motion_target_db=None,
            phone_templates_path=tpl,
        ),
        M1GridVariant(
            "m1_coart_100_corpus_tpl",
            trials=100,
            phone_templates_path=tpl_corpus,
        ),
    ]


def _ensure_corpus(
    variant: M1GridVariant,
    corpus_root: Path,
    *,
    force: bool = False,
) -> Path:
    out = corpus_root / variant.tag
    if not force and (out / "events.csv").is_file() and (out / "signals.npy").is_file():
        return out
    ds = build_gowda_dataset_config(
        out,
        n_trials=int(variant.trials),
        seed=int(variant.seed),
        realism=str(variant.realism),
        snr_target_db=variant.snr_target_db,
        snr_motion_target_db=variant.snr_motion_target_db,
        phone_templates_path=variant.phone_templates_path,
        coarticulation_enabled=bool(variant.coarticulation_enabled),
    )
    generate_dataset(ds)
    return out


def run_m1_transfer_grid(
    real_dir: Path,
    *,
    corpus_root: Path,
    variants: Optional[Sequence[M1GridVariant]] = None,
    pretrain_epochs: int = 30,
    anchor_epochs: int = 15,
    device_preferred: str = "auto",
    force_regen: bool = False,
    skip_transfer: bool = False,
    tags: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Generate corpora + sim-only/anchor transfer for each M1 grid variant."""
    real_dir = Path(real_dir)
    corpus_root = Path(corpus_root)
    corpus_root.mkdir(parents=True, exist_ok=True)
    all_variants = list(variants if variants is not None else default_m1_grid())
    if tags:
        tag_set = {str(t).strip() for t in tags}
        all_variants = [v for v in all_variants if v.tag in tag_set]

    report: Dict[str, Any] = {
        "real_dir": str(real_dir),
        "corpus_root": str(corpus_root),
        "variants": [],
    }
    for variant in all_variants:
        print(f"[openalterego] m1-grid variant={variant.tag}")
        sim_dir = _ensure_corpus(variant, corpus_root, force=bool(force_regen))
        row: Dict[str, Any] = {"variant": variant.to_dict(), "sim_dir": str(sim_dir)}
        if not skip_transfer:
            out_report = real_dir / "ablations" / f"sim_transfer_{variant.tag}.json"
            transfer = eval_sim_only_anchor(
                sim_dir,
                real_dir,
                pretrain_epochs=int(pretrain_epochs),
                anchor_epochs=int(anchor_epochs),
                seed=int(variant.seed),
                device_preferred=device_preferred,
            )
            out_report.parent.mkdir(parents=True, exist_ok=True)
            out_report.write_text(json.dumps(transfer, indent=2), encoding="utf-8")
            row["transfer_report"] = str(out_report)
            sim_only = next((r for r in transfer["runs"] if r["tag"] == "sim_only"), None)
            anchor = next((r for r in transfer["runs"] if r["tag"] == "sim_pretrain_real_anchor"), None)
            row["sim_only_wer"] = (sim_only or {}).get("real_test", {}).get("wer")
            row["anchor_wer"] = (anchor or {}).get("real_test", {}).get("wer")
            print(
                f"[openalterego] {variant.tag}: anchor WER={row['anchor_wer']} "
                f"sim_only={row['sim_only_wer']}"
            )
        report["variants"].append(row)

    report["variants"].sort(
        key=lambda r: float(r.get("anchor_wer") if r.get("anchor_wer") is not None else 1.0)
    )
    out = real_dir / "ablations" / "m1_transfer_grid_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[openalterego] m1-grid report -> {out}")
    return report
