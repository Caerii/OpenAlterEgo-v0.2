"""Unified CTC evaluation: greedy/beam decode, val/test splits."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..data_split import gowda_official_train_val_test_indices
from ..device import resolve_device
from ..phonology.gowda_lexicon import PHONEME_ALPHABET, ids_to_phonemes
from ..spd.basis import SPDBasis, ensure_gowda_spd_basis
from .dataset import PhonemeCTCDataset, ctc_collate_raw, ctc_collate_spd
from .decode import (
    beam_ctc_decode_batch,
    build_lexicon_prefix_set,
    greedy_ctc_decode,
    ids_to_phone_strings,
)
from .lexicon_viterbi import lexicon_viterbi_batch
from .model import GowdaCTCModel, GowdaSPDCTCModel, GowdaSPDCTCModelLegacy
from .metrics import match_word_from_phonemes, phoneme_error_rate, word_error_rate, word_from_phoneme_ids

from .util import input_lengths, unpack_batch

DecodeMode = Literal["greedy", "beam", "beam_lexicon", "lexicon_viterbi"]
SplitName = Literal["train", "val", "test"]


def _events_for_split(events: pd.DataFrame, split: SplitName) -> pd.DataFrame:
    trial_ids = events["trial_id"].astype(int).values
    tr_idx, va_idx, te_idx = gowda_official_train_val_test_indices(trial_ids)
    if split == "train":
        idx = tr_idx
    elif split == "val":
        idx = va_idx
    else:
        idx = te_idx
    return events.iloc[idx].reset_index(drop=True)


def build_ctc_model_from_checkpoint(ckpt: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    n_ph = int(ckpt.get("n_phonemes", len(PHONEME_ALPHABET)))
    ft = str(ckpt.get("feature_type", "raw"))
    if ft == "spd":
        sd = ckpt["state_dict"]
        if "input_norm.weight" in sd:
            model = GowdaSPDCTCModel(
                int(ckpt["feature_dim"]),
                n_ph,
                hidden=int(ckpt.get("hidden", 256)),
                num_layers=int(ckpt.get("num_layers", 3)),
                dropout=float(ckpt.get("dropout", 0.2)),
            )
        else:
            model = GowdaSPDCTCModelLegacy(
                int(ckpt["feature_dim"]),
                n_ph,
                hidden=int(ckpt.get("hidden", 256)),
                num_layers=int(ckpt.get("num_layers", 2)),
            )
    else:
        model = GowdaCTCModel(int(ckpt["channels"]), n_ph)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device)


def evaluate_ctc(
    model: torch.nn.Module,
    dataset: PhonemeCTCDataset,
    device: torch.device,
    *,
    batch_size: int = 32,
    decode_mode: DecodeMode = "greedy",
    beam_width: int = 50,
) -> Dict[str, Any]:
    feature_type = str(getattr(dataset, "feature_type", "raw"))
    collate_fn = ctc_collate_spd if feature_type == "spd" else ctc_collate_raw
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, collate_fn=collate_fn, num_workers=0)

    lex_prefixes = None
    if decode_mode == "beam_lexicon":
        lex_prefixes = build_lexicon_prefix_set(dataset.lexicon)

    hyp_ph: List[List[str]] = []
    ref_ph: List[List[str]] = []
    hyp_w: List[str] = []
    ref_w: List[str] = []
    offset = 0

    model.eval()
    with torch.no_grad():
        for batch in loader:
            x, targets, t_lens, x_lens = unpack_batch(batch)
            x = x.to(device)
            logits = model(x)
            in_lens = input_lengths(logits, x_lens)

            if decode_mode == "lexicon_viterbi":
                hyp_words_batch = lexicon_viterbi_batch(
                    logits.detach().cpu().numpy(),
                    in_lens.detach().cpu().numpy(),
                    dataset.lexicon,
                )
                batch_ids = beam_ctc_decode_batch(
                    logits, in_lens, beam_width=int(beam_width), lexicon_prefixes=None
                )
            elif decode_mode == "greedy":
                batch_ids = greedy_ctc_decode(logits, in_lens)
                hyp_words_batch = None
            else:
                batch_ids = beam_ctc_decode_batch(
                    logits,
                    in_lens,
                    beam_width=int(beam_width),
                    lexicon_prefixes=lex_prefixes,
                )
                hyp_words_batch = None

            hyp_batch = ids_to_phone_strings(batch_ids)
            for i in range(x.size(0)):
                idx = offset + i
                ref_w.append(dataset.word_labels[idx])
                ids = batch_ids[i]
                if decode_mode == "lexicon_viterbi":
                    hyp_w.append(hyp_words_batch[i] if hyp_words_batch else "")
                    hyp_ph.append(dataset.lexicon.get(hyp_words_batch[i], []))
                elif decode_mode == "beam_lexicon":
                    hyp_w.append(word_from_phoneme_ids(ids, dataset.lexicon))
                    hyp_ph.append(hyp_batch[i])
                else:
                    hyp_w.append(match_word_from_phonemes(hyp_batch[i], dataset.lexicon))
                    hyp_ph.append(hyp_batch[i])
                ref_ids = targets[i, : int(t_lens[i])].tolist()
                ref_ph.append(ids_to_phonemes(ref_ids))
            offset += x.size(0)

    per = phoneme_error_rate(hyp_ph, ref_ph)
    wer = word_error_rate(hyp_w, ref_w)
    acc = float(np.mean([h == r for h, r in zip(hyp_w, ref_w)])) if ref_w else 0.0
    return {
        "per": round(per, 4),
        "wer": round(wer, 4),
        "word_acc": round(acc, 4),
        "n": len(ref_w),
        "decode_mode": str(decode_mode),
        "beam_width": int(beam_width) if decode_mode != "greedy" else 0,
        "hyp_words": hyp_w,
        "ref_words": ref_w,
    }


def eval_checkpoint(
    checkpoint: Union[str, Path],
    data_dir: Union[str, Path],
    *,
    split: SplitName = "test",
    decode_mode: DecodeMode = "beam",
    beam_width: int = 50,
    batch_size: int = 32,
    device_preferred: str = "auto",
    use_upper_tri: bool = False,
    feature_mode: str = "full",
) -> Dict[str, Any]:
    data_dir = Path(data_dir)
    device = resolve_device(device_preferred)
    ckpt = torch.load(Path(checkpoint), map_location=device, weights_only=False)
    model = build_ctc_model_from_checkpoint(ckpt, device)

    signals = np.load(data_dir / "signals.npy", mmap_mode="r")
    events = pd.read_csv(data_dir / "events.csv")
    split_events = _events_for_split(events, split)

    ft = str(ckpt.get("feature_type", "raw"))
    spd_basis: Optional[SPDBasis] = None
    if ft == "spd":
        spd_basis = ensure_gowda_spd_basis(
            data_dir,
            fs_hz=int(ckpt.get("fs", 5000)),
            emg_mode=str(ckpt.get("emg_mode", "gowda")),
            seed=int(ckpt.get("seed", 1337)),
            use_upper_tri=bool(ckpt.get("use_upper_tri", use_upper_tri)),
            feature_mode=str(ckpt.get("feature_mode", feature_mode)),
        )

    ds = PhonemeCTCDataset(
        np.asarray(signals, dtype=np.float32),
        split_events,
        fs_hz=int(ckpt.get("fs", 5000)),
        segment_ms=int(ckpt.get("segment_ms", 2000)),
        seed=int(ckpt.get("seed", 1337)),
        emg_mode=str(ckpt.get("emg_mode", "gowda")),
        per_event_preprocess=True,
        feature_type=ft,  # type: ignore[arg-type]
        spd_basis=spd_basis,
        session_dir=str(data_dir),
        use_upper_tri=bool(ckpt.get("use_upper_tri", use_upper_tri)),
        feature_mode=str(ckpt.get("feature_mode", feature_mode)),
    )

    metrics = evaluate_ctc(
        model,
        ds,
        device,
        batch_size=int(batch_size),
        decode_mode=decode_mode,
        beam_width=int(beam_width),
    )
    return {
        "split": split,
        "checkpoint": str(checkpoint),
        **{k: v for k, v in metrics.items() if k not in ("hyp_words", "ref_words")},
        "hyp_words": metrics["hyp_words"],
        "ref_words": metrics["ref_words"],
    }
