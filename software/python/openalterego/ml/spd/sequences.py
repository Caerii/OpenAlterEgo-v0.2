"""Build σ(τ) sequences for CTC training."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .basis import SPDBasis
from .features import segment_to_sigma_sequence


def build_spd_segment_sequences(
    signals: np.ndarray,
    events: pd.DataFrame,
    label_to_id: Dict[str, int],
    basis: SPDBasis,
    *,
    fs_hz: int,
    emg_mode: str = "gowda",
    per_event_preprocess: bool = True,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Per-event σ(τ) sequences and parallel int labels."""
    from ...dsp.filters import preprocess_basic

    seqs: List[np.ndarray] = []
    ys: List[int] = []
    for _, row in events.iterrows():
        s, e = int(row["start_sample"]), int(row["end_sample"])
        lab = str(row["label"])
        if lab not in label_to_id:
            continue
        seg = np.asarray(signals[s:e, :], dtype=np.float32)
        if per_event_preprocess:
            seg = preprocess_basic(
                seg,
                fs_hz=int(fs_hz),
                mode=emg_mode,  # type: ignore[arg-type]
                rectify_signals=False,
                normalize_mode="zscore",
            )
        if seg.shape[0] < 8:
            continue
        sig = segment_to_sigma_sequence(
            seg,
            basis.basis_q,
            fs_hz=int(fs_hz),
            window_ms=int(basis.window_ms),
            step_ms=int(basis.step_ms),
            eta=float(basis.eta),
            use_upper_tri=bool(basis.use_upper_tri),
            feature_mode=str(basis.feature_mode),
        )
        seqs.append(sig)
        ys.append(int(label_to_id[lab]))
    if not seqs:
        d = int(basis.feature_dim)
        return [], np.zeros((0,), dtype=np.int64)
    return seqs, np.asarray(ys, dtype=np.int64)
