"""Train/validation splits for event tables (stratified by label)."""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np


def stratified_train_val_indices(
    labels_per_event: np.ndarray,
    val_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split *row indices* into train and val so each class contributes to val when possible.

    Parameters
    ----------
    labels_per_event:
        Shape ``(n_events,)`` — one label string (or hashable) per `events.csv` row.
    val_fraction:
        Target fraction of rows in validation (0 disables val).
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    train_idx, val_idx
        Integer index arrays into the original events table (possibly empty val).
    """
    n = int(len(labels_per_event))
    if n == 0:
        return np.array([], dtype=np.intp), np.array([], dtype=np.intp)
    if val_fraction <= 0.0:
        return np.arange(n, dtype=np.intp), np.array([], dtype=np.intp)

    rng = np.random.default_rng(int(seed))
    tr_parts: list[np.ndarray] = []
    va_parts: list[np.ndarray] = []
    frac = float(val_fraction)

    for lab in np.unique(labels_per_event):
        idx = np.flatnonzero(labels_per_event == lab)
        rng.shuffle(idx)
        n_i = int(len(idx))
        if n_i == 0:
            continue
        n_val = int(round(n_i * frac))
        if n_i == 1:
            n_val = 0
        else:
            n_val = min(max(1, n_val), n_i - 1)
        va_parts.append(idx[:n_val])
        tr_parts.append(idx[n_val:])

    tr = np.concatenate(tr_parts) if tr_parts else np.array([], dtype=np.intp)
    va = np.concatenate(va_parts) if va_parts else np.array([], dtype=np.intp)
    rng.shuffle(tr)
    rng.shuffle(va)
    return tr, va


def stratified_group_train_val_indices(
    labels_per_event: np.ndarray,
    groups: np.ndarray,
    val_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split by group id (e.g. ``trial_id``) so no group appears in both train and val.

    Groups are assigned to val with stratification on each group's dominant label.
    """
    labels_per_event = np.asarray(labels_per_event)
    groups = np.asarray(groups)
    if len(labels_per_event) != len(groups):
        raise ValueError("labels and groups must have the same length")
    n = int(len(labels_per_event))
    if n == 0:
        return np.array([], dtype=np.intp), np.array([], dtype=np.intp)
    if val_fraction <= 0.0:
        return np.arange(n, dtype=np.intp), np.array([], dtype=np.intp)

    unique_groups = np.unique(groups)
    group_labels: list[str] = []
    for g in unique_groups:
        mask = groups == g
        labs, counts = np.unique(labels_per_event[mask], return_counts=True)
        group_labels.append(str(labs[int(np.argmax(counts))]))

    tr_gi, va_gi = stratified_train_val_indices(np.array(group_labels), val_fraction, seed)
    tr_groups = set(unique_groups[tr_gi].tolist())
    va_groups = set(unique_groups[va_gi].tolist())
    tr_idx = np.flatnonzero(np.isin(groups, list(tr_groups)))
    va_idx = np.flatnonzero(np.isin(groups, list(va_groups)))
    return tr_idx, va_idx


def gowda_sentence_train_val_indices(
    trial_ids: np.ndarray,
    *,
    n_train_sentences: int = 370,
    n_val_sentences: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """Alias for :func:`gowda_official_train_val_indices` (emg2speech notebook split)."""
    return gowda_official_train_val_indices(
        trial_ids,
        train_trials=int(n_train_sentences),
        val_trials=int(n_val_sentences),
    )


def gowda_official_train_val_indices(
    trial_ids: np.ndarray,
    *,
    train_trials: int = 370,
    val_trials: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """Official emg2speech small-vocab split: 370 train / 30 val sentences (test held out).

    Events with ``trial_id >= train_trials + val_trials`` are excluded from both sets.
    """
    tr_idx, va_idx, _ = gowda_official_train_val_test_indices(
        trial_ids,
        train_trials=train_trials,
        val_trials=val_trials,
    )
    return tr_idx, va_idx


def gowda_official_train_val_test_indices(
    trial_ids: np.ndarray,
    *,
    train_trials: int = 370,
    val_trials: int = 30,
    test_trials: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Official emg2speech small-vocab split: 370 / 30 / 100 sentences (trials)."""
    trial_ids = np.asarray(trial_ids, dtype=np.intp)
    n_train = int(train_trials)
    n_val = int(val_trials)
    n_test = int(test_trials)
    tr_idx = np.flatnonzero(trial_ids < n_train)
    va_idx = np.flatnonzero((trial_ids >= n_train) & (trial_ids < n_train + n_val))
    te_idx = np.flatnonzero(
        (trial_ids >= n_train + n_val) & (trial_ids < n_train + n_val + n_test)
    )
    return tr_idx, va_idx, te_idx


def gowda_sim_transfer_merged_indices(
    trial_ids: np.ndarray,
    *,
    sim_trials: int = 500,
    train_trials: int = 370,
    val_trials: int = 30,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split for sim+real merged sessions: sim uses official 370/30/100; real block (id >= sim_trials) is train-only."""
    trial_ids = np.asarray(trial_ids, dtype=np.intp)
    n_train = int(train_trials)
    n_val = int(val_trials)
    sim_n = int(sim_trials)
    tr_idx = np.flatnonzero((trial_ids < n_train) | (trial_ids >= sim_n))
    va_idx = np.flatnonzero((trial_ids >= n_train) & (trial_ids < n_train + n_val))
    te_idx = np.flatnonzero((trial_ids >= n_train + n_val) & (trial_ids < sim_n))
    return tr_idx, va_idx, te_idx


def resolve_gowda_train_val_test_indices(
    trial_ids: np.ndarray,
    *,
    split_mode: str = "",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pick official or sim-transfer merged split."""
    mode = str(split_mode or "").strip().lower()
    tmax = int(np.max(trial_ids)) if len(trial_ids) else 0
    if mode == "sim_transfer_merged" or tmax >= 500:
        return gowda_sim_transfer_merged_indices(trial_ids)
    return gowda_official_train_val_test_indices(trial_ids)


def resolve_train_val_indices(
    events_labels: np.ndarray,
    val_fraction: float,
    seed: int,
    *,
    split_by: str = "auto",
    groups: Union[np.ndarray, None] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pick event- or group-level stratified split."""
    mode = str(split_by or "auto").strip().lower()
    if mode == "auto":
        mode = "group" if groups is not None and len(groups) == len(events_labels) else "event"
    if mode == "group":
        if groups is None:
            raise ValueError("split_by=group requires groups array")
        return stratified_group_train_val_indices(events_labels, groups, val_fraction, seed)
    if mode == "gowda":
        if groups is None:
            raise ValueError("split_by=gowda requires trial_id groups array")
        return gowda_official_train_val_indices(groups)
    if mode == "event":
        return stratified_train_val_indices(events_labels, val_fraction, seed)
    raise ValueError(f"unknown split_by: {split_by!r} (use auto, event, group, or gowda)")
