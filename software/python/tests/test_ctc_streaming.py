"""Tests for PTT streaming CTC decode."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np


class TestUtteranceSegmenter(unittest.TestCase):
    def test_ptt_buffer_roundtrip(self) -> None:
        from openalterego.runtime.utterance_segmenter import UtteranceSegmenter, UtteranceSegmenterConfig

        fs = 5000
        ch = 31
        seg = UtteranceSegmenter(UtteranceSegmenterConfig(fs_hz=fs, channels=ch, min_utterance_ms=100))
        chunk = np.zeros((250, ch), dtype=np.float32)
        seg.feed(chunk)
        seg.on_ptt_start()
        for _ in range(4):
            seg.feed(chunk)
        utt = seg.on_ptt_end()
        self.assertIsNotNone(utt)
        assert utt is not None
        self.assertEqual(utt.shape[1], ch)
        self.assertGreaterEqual(utt.shape[0], int(0.1 * fs))


class TestStreamingCTCDecoder(unittest.TestCase):
    def test_finalize_real_trial_word_if_checkpoint_present(self) -> None:
        ckpt = Path("sessions/gowda_sv_full/ablations/ctc_spd_v3_diag_delta_seed1337.pt")
        sess = Path("sessions/gowda_sv_full")
        if not ckpt.is_file() or not (sess / "signals.npy").is_file():
            self.skipTest("Gowda session + phase5 checkpoint not present")

        import pandas as pd

        from openalterego.runtime.ctc_streaming import build_streaming_ctc_decoder

        ev = pd.read_csv(sess / "events.csv")
        row = ev.iloc[0]
        sig = np.load(sess / "signals.npy", mmap_mode="r")
        s0, s1 = int(row["start_sample"]), int(row["end_sample"])
        utt = np.asarray(sig[s0:s1], dtype=np.float32)

        dec = build_streaming_ctc_decoder(
            ckpt,
            sess,
            device_preferred="cpu",
            decode_mode="trial",
        )
        out = dec.finalize_utterance(utt)
        self.assertTrue(out.text)
        self.assertEqual(out.words, [str(row["label"])])


if __name__ == "__main__":
    unittest.main()
