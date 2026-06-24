"""Train GRU+CTC on Gowda word events (raw EMG CNN or SPD sigma(tau))."""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data_split import resolve_gowda_train_val_test_indices
from ..datasets.events import load_gowda_events, read_split_mode
from ..device import configure_cuda_for_training, resolve_device
from ..phonology.gowda_lexicon import PHONEME_ALPHABET
from ..spd.basis import ensure_gowda_spd_basis
from ..training_perf import resolve_use_amp
from .dataset import FeatureType, PhonemeCTCDataset, ctc_collate_raw, ctc_collate_spd
from .eval import DecodeMode, evaluate_ctc
from .model import GowdaCTCModel, GowdaSPDCTCModel

from .util import input_lengths, unpack_batch


def train_gowda_ctc(
    data_dir: Path,
    *,
    fs_hz: int = 5000,
    segment_ms: int = 2000,
    emg_mode: str = "gowda",
    feature_type: FeatureType = "raw",
    use_upper_tri: bool = False,
    feature_mode: str = "full",
    augment_train: bool = False,
    hidden: int = 256,
    num_layers: int = 3,
    dropout: float = 0.2,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    warmup_epochs: int = 0,
    seed: int = 1337,
    device_preferred: str = "auto",
    save_path: Optional[Path] = None,
    basis_session_dir: Optional[Path] = None,
    init_checkpoint: Optional[Path] = None,
    val_decode_mode: DecodeMode = "greedy",
    beam_width: int = 50,
    eval_test: bool = True,
    log_every: int = 5,
) -> Dict[str, Any]:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    device = resolve_device(device_preferred)
    configure_cuda_for_training(device)
    use_amp = resolve_use_amp(device, no_amp=False)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    data_dir = Path(data_dir)
    init_ckpt: Optional[Dict[str, Any]] = None
    if init_checkpoint is not None and Path(init_checkpoint).is_file():
        init_ckpt = torch.load(Path(init_checkpoint), map_location="cpu", weights_only=False)
        hidden = int(init_ckpt.get("hidden", hidden))
        num_layers = int(init_ckpt.get("num_layers", num_layers))
        dropout = float(init_ckpt.get("dropout", dropout))

    signals = np.load(data_dir / "signals.npy", mmap_mode="r")
    events = load_gowda_events(data_dir)
    if events.empty:
        raise ValueError(f"no trial events in {data_dir / 'events.csv'}")
    trial_ids = events["trial_id"].astype(int).values
    tr_idx, val_idx, te_idx = resolve_gowda_train_val_test_indices(
        trial_ids, split_mode=read_split_mode(data_dir)
    )
    tr_events = events.iloc[tr_idx].reset_index(drop=True)
    val_events = events.iloc[val_idx].reset_index(drop=True)
    te_events = events.iloc[te_idx].reset_index(drop=True)

    spd_basis = None
    if str(feature_type) == "spd":
        spd_basis = ensure_gowda_spd_basis(
            data_dir,
            basis_session_dir=basis_session_dir,
            fs_hz=int(fs_hz),
            emg_mode=str(emg_mode),
            seed=int(seed),
            use_upper_tri=bool(use_upper_tri),
            feature_mode=str(feature_mode),
        )

    common_ds = dict(
        fs_hz=int(fs_hz),
        segment_ms=int(segment_ms),
        emg_mode=str(emg_mode),
        per_event_preprocess=True,
        feature_type=str(feature_type),  # type: ignore[arg-type]
        spd_basis=spd_basis,
        session_dir=str(data_dir),
        use_upper_tri=bool(use_upper_tri),
        feature_mode=str(feature_mode),
    )
    sig = np.asarray(signals, dtype=np.float32)
    ds_tr = PhonemeCTCDataset(sig, tr_events, seed=int(seed), augment_train=bool(augment_train), **common_ds)
    ds_val = PhonemeCTCDataset(sig, val_events, seed=int(seed) + 1, **common_ds)
    ds_te = PhonemeCTCDataset(sig, te_events, seed=int(seed) + 2, **common_ds) if len(te_events) else None

    n_ph = len(PHONEME_ALPHABET)
    if str(feature_type) == "spd":
        assert spd_basis is not None
        model: torch.nn.Module = GowdaSPDCTCModel(
            int(spd_basis.feature_dim),
            n_ph,
            hidden=int(hidden),
            num_layers=int(num_layers),
            dropout=float(dropout),
        ).to(device)
        n_ch = int(spd_basis.channels)
        feature_dim = int(spd_basis.feature_dim)
    else:
        n_ch = int(ds_tr.X.shape[1])  # type: ignore[union-attr]
        feature_dim = None
        model = GowdaCTCModel(n_ch, n_ph).to(device)

    if init_ckpt is not None and "state_dict" in init_ckpt:
        model.load_state_dict(init_ckpt["state_dict"], strict=False)

    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    warmup = max(0, int(warmup_epochs))
    total = max(1, int(epochs))

    def _lr_lambda(ep: int) -> float:
        if warmup > 0 and ep < warmup:
            return float(ep + 1) / float(warmup)
        if total <= warmup:
            return 1.0
        progress = (ep - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + float(np.cos(np.pi * progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_lambda)
    loss_fn = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    collate_fn = ctc_collate_spd if str(feature_type) == "spd" else ctc_collate_raw
    dl_tr = DataLoader(ds_tr, batch_size=int(batch_size), shuffle=True, collate_fn=collate_fn, num_workers=0)

    tag = "ctc_spd_v3" if str(feature_type) == "spd" and str(feature_mode) != "full" else (
        "ctc_spd_v2" if str(feature_type) == "spd" else "ctc_gowda"
    )
    ckpt_path = save_path or (data_dir / "ablations" / f"{tag}_seed{seed}.pt")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    best_wer = 1e9
    t0 = time.perf_counter()

    for _epoch in range(int(epochs)):
        model.train()
        for batch in dl_tr:
            x, targets, t_lens, x_lens = unpack_batch(batch)
            x = x.to(device, non_blocking=True)
            targets = targets.to(device)
            t_lens = t_lens.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x)
                log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
                in_lens = input_lengths(logits, x_lens.to(device) if x_lens is not None else None)
                flat_targets = []
                for i in range(targets.size(0)):
                    flat_targets.extend(targets[i, : int(t_lens[i])].tolist())
                target_tensor = torch.tensor(flat_targets, dtype=torch.long, device=device)
                loss = loss_fn(log_probs, target_tensor, in_lens, t_lens)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()

        val_m = evaluate_ctc(
            model,
            ds_val,
            device,
            batch_size=int(batch_size),
            decode_mode=str(val_decode_mode),  # type: ignore[arg-type]
            beam_width=int(beam_width),
        )
        ep = _epoch + 1
        if ep == 1 or ep % max(1, int(log_every)) == 0 or ep == int(epochs):
            print(
                f"[openalterego] epoch {ep}/{epochs} "
                f"val_per={val_m['per']:.3f} val_wer={val_m['wer']:.3f} "
                f"decode={val_decode_mode}"
            )
        if float(val_m["wer"]) <= best_wer:
            best_wer = float(val_m["wer"])
            ckpt_payload: Dict[str, Any] = {
                "state_dict": model.state_dict(),
                "fs": int(fs_hz),
                "channels": n_ch,
                "segment_ms": int(segment_ms),
                "emg_mode": str(emg_mode),
                "feature_type": str(feature_type),
                "use_upper_tri": bool(use_upper_tri),
                "feature_mode": str(feature_mode),
                "hidden": int(hidden),
                "num_layers": int(num_layers),
                "dropout": float(dropout),
                "n_phonemes": n_ph,
                "seed": int(seed),
                "val": {k: v for k, v in val_m.items() if k not in ("hyp_words", "ref_words")},
            }
            if feature_dim is not None:
                ckpt_payload["feature_dim"] = int(feature_dim)
            torch.save(ckpt_payload, ckpt_path)

    elapsed = time.perf_counter() - t0
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])

    train_m = evaluate_ctc(model, ds_tr, device, batch_size=int(batch_size), decode_mode="greedy")
    val_m = evaluate_ctc(
        model,
        ds_val,
        device,
        batch_size=int(batch_size),
        decode_mode=str(val_decode_mode),  # type: ignore[arg-type]
        beam_width=int(beam_width),
    )
    val_beam_lex = evaluate_ctc(
        model, ds_val, device, batch_size=int(batch_size), decode_mode="beam_lexicon", beam_width=int(beam_width)
    )

    out: Dict[str, Any] = {
        "seed": int(seed),
        "feature_type": str(feature_type),
        "use_upper_tri": bool(use_upper_tri),
        "feature_mode": str(feature_mode),
        "n_train": len(ds_tr),
        "n_val": len(ds_val),
        "n_test": len(te_events) if ds_te is not None else 0,
        "n_phonemes": n_ph,
        "epochs": int(epochs),
        "elapsed_s": round(float(elapsed), 1),
        "checkpoint": str(ckpt_path),
        "train": {k: v for k, v in train_m.items() if k not in ("hyp_words", "ref_words")},
        "val": {k: v for k, v in val_m.items() if k not in ("hyp_words", "ref_words")},
        "val_beam_lexicon": {k: v for k, v in val_beam_lex.items() if k not in ("hyp_words", "ref_words")},
        "best_val_wer_at_save": round(float(best_wer), 4),
        "hyp_words_val": val_m.get("hyp_words", []),
        "ref_words_val": val_m.get("ref_words", []),
    }

    if eval_test and ds_te is not None and len(ds_te) > 0:
        for mode in ("greedy", "beam", "beam_lexicon", "lexicon_viterbi"):
            te_m = evaluate_ctc(
                model,
                ds_te,
                device,
                batch_size=int(batch_size),
                decode_mode=mode,  # type: ignore[arg-type]
                beam_width=int(beam_width),
            )
            out[f"test_{mode}"] = {k: v for k, v in te_m.items() if k not in ("hyp_words", "ref_words")}
            if mode == "beam":
                out["hyp_words_test"] = te_m.get("hyp_words", [])
                out["ref_words_test"] = te_m.get("ref_words", [])

    if spd_basis is not None:
        out["spd_basis"] = spd_basis.to_dict()
    return out
