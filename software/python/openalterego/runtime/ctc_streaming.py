"""Streaming open-speech CTC decode (PTT finalize path)."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..ml.ctc.infer import LoadedCTCModel, forward_log_probs, load_ctc_model
from ..ml.ctc.lexicon_viterbi import lexicon_viterbi_word
from ..ml.ctc.trial_lm import TrialLanguageModel, fit_trial_lm
from ..ml.phonology.gowda_lexicon import build_lexicon
from ..ml.spd.online import OnlineSPDStream
from .utterance_segmenter import UtteranceSegmenter, UtteranceSegmenterConfig


@dataclass
class CTCStreamConfig:
    fs_hz: int
    channels: int
    pad_ms: int = 150
    min_utterance_ms: int = 200
    max_utterance_ms: int = 15000
    feature_mode: str = "diag_delta"
    emg_mode: str = "gowda"
    lm_weight: float = 0.0
    decode_mode: str = "word"  # word | trial


@dataclass
class FinalTranscript:
    utterance_id: str
    text: str
    words: List[str]
    confidence: float
    meta: Dict[str, Any]


class StreamingCTCDecoder:
    """PTT-buffered SPD+CTC decode for open speech."""

    def __init__(
        self,
        loaded: LoadedCTCModel,
        lexicon: Dict[str, List[str]],
        lm: TrialLanguageModel,
        *,
        cfg: CTCStreamConfig,
        source_name: str = "unknown",
    ):
        self.loaded = loaded
        self.lexicon = lexicon
        self.lm = lm
        self.cfg = cfg
        self.source_name = str(source_name)
        self.segmenter = UtteranceSegmenter(
            UtteranceSegmenterConfig(
                fs_hz=int(cfg.fs_hz),
                channels=int(cfg.channels),
                pad_ms=int(cfg.pad_ms),
                min_utterance_ms=int(cfg.min_utterance_ms),
                max_utterance_ms=int(cfg.max_utterance_ms),
            )
        )
        self._trial_prev: List[str] = []
        self._spd = OnlineSPDStream(
            loaded.basis_q,
            fs_hz=int(cfg.fs_hz),
            channels=int(cfg.channels),
            feature_mode=str(cfg.feature_mode),
            emg_mode=str(cfg.emg_mode),
        )

    def on_ptt_start(self) -> None:
        self.segmenter.on_ptt_start()

    def feed_chunk(self, samples: np.ndarray) -> None:
        self.segmenter.feed(samples)

    def on_ptt_end(self) -> Optional[FinalTranscript]:
        utt = self.segmenter.on_ptt_end()
        if utt is None:
            return None
        return self.finalize_utterance(utt)

    def finalize_utterance(self, utterance: np.ndarray) -> FinalTranscript:
        t0 = time.time()
        uid = str(uuid.uuid4())
        sigma = self._spd.build_sequence(utterance)
        lp = forward_log_probs(self.loaded, sigma)

        if str(self.cfg.decode_mode) == "trial":
            # Single-word slot in ongoing trial (word_idx from len(prev) mod 4)
            wi = len(self._trial_prev) % 4
            from ..ml.ctc.trial_decode import decode_word_trial_lm

            word = decode_word_trial_lm(
                lp,
                int(lp.shape[0]),
                self.lexicon,
                self.lm,
                word_idx=wi,
                prev_words=list(self._trial_prev[-3:]),
                lm_weight=float(self.cfg.lm_weight),
            )
            self._trial_prev.append(word)
            if len(self._trial_prev) > 16:
                self._trial_prev = self._trial_prev[-12:]
            words = [word]
            score = 0.0
        else:
            word, score = lexicon_viterbi_word(lp, int(lp.shape[0]), self.lexicon)
            words = [word]

        text = " ".join(words)
        conf = float(min(1.0, max(0.0, 1.0 + float(score) / 80.0))) if words else 0.0
        return FinalTranscript(
            utterance_id=uid,
            text=text,
            words=words,
            confidence=conf,
            meta={
                "latency_ms": round((time.time() - t0) * 1000.0, 2),
                "n_samples": int(utterance.shape[0]),
                "n_sigma_frames": int(sigma.shape[0]),
                "source": self.source_name,
            },
        )


def build_streaming_ctc_decoder(
    checkpoint: Path | str,
    session_dir: Path | str,
    *,
    device_preferred: str = "auto",
    fs_hz: int = 5000,
    channels: int = 31,
    feature_mode: str = "diag_delta",
    decode_mode: str = "trial",
    lm_weight: float = 0.0,
    source_name: str = "unknown",
) -> StreamingCTCDecoder:
    session_dir = Path(session_dir)
    loaded = load_ctc_model(checkpoint, device_preferred=device_preferred, session_dir=session_dir)
    events_path = session_dir / "events.csv"
    if not events_path.is_file():
        raise FileNotFoundError(f"session events required for lexicon/LM: {events_path}")
    import pandas as pd

    events = pd.read_csv(events_path)
    labels = sorted({str(x) for x in events["label"].unique()})
    lm = fit_trial_lm(events)
    lexicon = build_lexicon(labels)
    cfg = CTCStreamConfig(
        fs_hz=int(fs_hz),
        channels=int(channels),
        feature_mode=str(feature_mode),
        emg_mode=str(loaded.emg_mode),
        lm_weight=float(lm_weight),
        decode_mode=str(decode_mode),
    )
    return StreamingCTCDecoder(loaded, lexicon, lm, cfg=cfg, source_name=source_name)
