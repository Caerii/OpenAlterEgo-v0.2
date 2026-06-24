"""CTC dataset: per-word EMG → phoneme id sequence (raw CNN or SPD σ(τ))."""

from __future__ import annotations

from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..phonology.gowda_lexicon import build_lexicon, phonemes_to_ids
from ..segment_cache import build_segment_arrays
from ..spd.basis import SPDBasis, ensure_gowda_spd_basis
from ..spd.sequence_cache import load_or_build_spd_sequences
from ..spd.sequences import build_spd_segment_sequences
from .augment import augment_sigma_sequence

FeatureType = Literal["raw", "spd"]


class PhonemeCTCDataset(Dataset):
    def __init__(
        self,
        signals: np.ndarray,
        events: pd.DataFrame,
        *,
        fs_hz: int,
        segment_ms: int,
        seed: int,
        emg_mode: str = "gowda",
        per_event_preprocess: bool = True,
        feature_type: FeatureType = "raw",
        spd_basis: Optional[SPDBasis] = None,
        session_dir: Optional[str] = None,
        use_upper_tri: bool = False,
        feature_mode: str = "full",
        augment_train: bool = False,
    ):
        self._augment_train = bool(augment_train)
        self._rng = np.random.default_rng(int(seed))
        words = sorted({str(x) for x in events["label"].unique()})
        self.lexicon = build_lexicon(words)
        label_to_id = {w: i for i, w in enumerate(words)}
        self.feature_type: FeatureType = str(feature_type)  # type: ignore[assignment]

        if self.feature_type == "spd":
            basis = spd_basis
            if basis is None:
                if session_dir is None:
                    raise ValueError("SPD features require spd_basis or session_dir")
                basis = ensure_gowda_spd_basis(
                    session_dir,
                    fs_hz=int(fs_hz),
                    emg_mode=str(emg_mode),
                    seed=int(seed),
                    use_upper_tri=bool(use_upper_tri),
                    feature_mode=str(feature_mode),
                )
            self.spd_basis = basis
            if session_dir is not None:
                seqs, y_word = load_or_build_spd_sequences(
                    signals,
                    events,
                    label_to_id,
                    basis,
                    session_dir,
                    split_tag=f"events_{len(events)}",
                    fs_hz=int(fs_hz),
                    emg_mode=str(emg_mode),
                    per_event_preprocess=bool(per_event_preprocess),
                )
            else:
                seqs, y_word = build_spd_segment_sequences(
                    signals,
                    events,
                    label_to_id,
                    basis,
                    fs_hz=int(fs_hz),
                    emg_mode=str(emg_mode),
                    per_event_preprocess=bool(per_event_preprocess),
                )
            self.X_spd: List[np.ndarray] = seqs
            self.X: Optional[np.ndarray] = None
        else:
            self.spd_basis = None
            X, y_word = build_segment_arrays(
                signals,
                events,
                label_to_id,
                fs_hz=int(fs_hz),
                segment_ms=int(segment_ms),
                seed=int(seed),
                per_event_preprocess=bool(per_event_preprocess),
                preprocess_emg_mode=str(emg_mode),
            )
            self.X = X
            self.X_spd = []

        self.phoneme_seqs: List[List[int]] = []
        self.word_labels: List[str] = []
        inv = {i: w for w, i in label_to_id.items()}
        for yi in y_word:
            word = inv[int(yi)]
            self.word_labels.append(word)
            self.phoneme_seqs.append(phonemes_to_ids(self.lexicon[word]))

        self.trial_ids = [int(x) for x in events["trial_id"].values]
        self.word_indices = [int(x) for x in events["word_idx"].values]

        n_events = len(self.X_spd) if self.feature_type == "spd" else int(len(self.X or []))
        if n_events != len(self.phoneme_seqs):
            raise ValueError("segment / phoneme length mismatch")

    def __len__(self) -> int:
        if self.feature_type == "spd":
            return int(len(self.X_spd))
        return int(len(self.X or []))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        if self.feature_type == "spd":
            x = np.asarray(self.X_spd[idx], dtype=np.float32)
            if self._augment_train:
                x = augment_sigma_sequence(x, self._rng)
            x_t = torch.from_numpy(x)
        else:
            x_t = torch.from_numpy(self.X[idx])  # type: ignore[index]
        ph = torch.tensor(self.phoneme_seqs[idx], dtype=torch.long)
        return x_t, ph, int(len(ph))


def ctc_collate_raw(batch: List[Tuple[torch.Tensor, torch.Tensor, int]]):
    xs, phs, ph_lens = zip(*batch)
    x = torch.stack(xs, dim=0)
    ph_lens_t = torch.tensor(ph_lens, dtype=torch.long)
    max_ph = max(int(l) for l in ph_lens) if ph_lens else 1
    targets = torch.zeros(len(batch), max_ph, dtype=torch.long)
    for i, ph in enumerate(phs):
        targets[i, : ph.shape[0]] = ph
    return x, targets, ph_lens_t


def ctc_collate(batch: List[Tuple[torch.Tensor, torch.Tensor, int]]):
    """Default raw EMG collate (B, C, T)."""
    return ctc_collate_raw(batch)


def ctc_collate_spd(batch: List[Tuple[torch.Tensor, torch.Tensor, int]]):
    xs, phs, ph_lens = zip(*batch)
    t_max = max(int(x.shape[0]) for x in xs)
    d = int(xs[0].shape[1])
    x_pad = torch.zeros(len(batch), t_max, d, dtype=xs[0].dtype)
    x_lens = torch.tensor([int(x.shape[0]) for x in xs], dtype=torch.long)
    for i, x in enumerate(xs):
        x_pad[i, : x.shape[0], :] = x
    ph_lens_t = torch.tensor(ph_lens, dtype=torch.long)
    max_ph = max(int(l) for l in ph_lens) if ph_lens else 1
    targets = torch.zeros(len(batch), max_ph, dtype=torch.long)
    for i, ph in enumerate(phs):
        targets[i, : ph.shape[0]] = ph
    return x_pad, targets, ph_lens_t, x_lens
