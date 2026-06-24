"""Offline utterance / trial decode using SPD+CTC."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..phonology.gowda_lexicon import build_lexicon
from ..spd.online import OnlineSPDStream
from .infer import LoadedCTCModel, forward_log_probs, load_ctc_model
from .trial_decode import decode_word_trial_lm
from .trial_lm import TrialLanguageModel, fit_trial_lm


def decode_trial_words(
    loaded: LoadedCTCModel,
    signals: np.ndarray,
    trial_events: pd.DataFrame,
    lexicon: Dict[str, List[str]],
    lm: TrialLanguageModel,
    *,
    lm_weight: float = 0.0,
) -> Dict[str, Any]:
    """Decode one Gowda trial (4 words) in trial order."""
    if loaded.basis_q is None:
        raise ValueError("SPD basis required for decode_trial_words")

    ev = trial_events.sort_values("word_idx").reset_index(drop=True)
    stream = OnlineSPDStream(
        loaded.basis_q,
        fs_hz=loaded.fs_hz,
        channels=loaded.channels,
        feature_mode=loaded.feature_mode,
        emg_mode=loaded.emg_mode,
    )

    hyps: List[str] = []
    refs: List[str] = []
    prev: List[str] = []
    details: List[Dict[str, Any]] = []

    for _, row in ev.iterrows():
        s, e = int(row["start_sample"]), int(row["end_sample"])
        ref = str(row["label"])
        seg = np.asarray(signals[s:e, :], dtype=np.float32)
        sigma = stream.build_sequence(seg)
        lp = forward_log_probs(loaded, sigma)
        wi = int(row["word_idx"])
        pred = decode_word_trial_lm(
            lp,
            int(lp.shape[0]),
            lexicon,
            lm,
            word_idx=wi,
            prev_words=prev,
            lm_weight=float(lm_weight),
        )
        hyps.append(pred)
        refs.append(ref)
        prev.append(pred)
        details.append({"word_idx": wi, "ref": ref, "hyp": pred, "correct": pred == ref})

    acc = float(np.mean([h == r for h, r in zip(hyps, refs)])) if refs else 0.0
    return {
        "trial_id": int(ev["trial_id"].iloc[0]) if "trial_id" in ev.columns else -1,
        "words": details,
        "hyp_words": hyps,
        "ref_words": refs,
        "word_acc": round(acc, 4),
        "n_words": len(refs),
    }


def decode_session_trial(
    session_dir: Path,
    trial_id: int,
    checkpoint: Path,
    *,
    device_preferred: str = "auto",
    lm_weight: float = 0.0,
) -> Dict[str, Any]:
    """Decode all words in one trial from a session folder."""
    session_dir = Path(session_dir)
    loaded = load_ctc_model(checkpoint, device_preferred=device_preferred, session_dir=session_dir)
    events = pd.read_csv(session_dir / "events.csv")
    if "trial_id" not in events.columns:
        raise ValueError("events.csv missing trial_id")
    trial_ev = events[events["trial_id"].astype(int) == int(trial_id)].copy()
    if trial_ev.empty:
        raise ValueError(f"trial_id {trial_id} not found in session")

    signals = np.load(session_dir / "signals.npy", mmap_mode="r")
    labels = sorted({str(x) for x in events["label"].unique()})
    lexicon = build_lexicon(labels)
    lm = fit_trial_lm(events)

    result = decode_trial_words(
        loaded,
        np.asarray(signals, dtype=np.float32),
        trial_ev,
        lexicon,
        lm,
        lm_weight=float(lm_weight),
    )
    result["session"] = str(session_dir)
    result["checkpoint"] = str(checkpoint)
    return result
