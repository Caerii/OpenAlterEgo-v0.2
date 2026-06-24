# Dataset: Large-Vocabulary Corpus (§4)

Paper: [arXiv:2502.05762v2](https://arxiv.org/html/2502.05762v2) Section 4

---

## Summary

| Field | Value |
|-------|-------|
| Source sentences | Adapted from Willett et al. BCI speech corpus |
| Unique words | ~**6 500** |
| Sentences | ~**11 000** total; split **8000 / 1000 / 1970** |
| Modality | **Silent EMG only** (no audible speech, no audio targets) |
| Rate | ~160 words/min |
| Channels | 31 @ 5 kHz |

---

## OSF assets

Typically `dataLargeVocab.pkl` / `labelsLargeVocab.pkl` in emg2speech DATA box — **not** yet imported by OpenAlterEgo.

---

## Reported metrics (σ(τ) + 3-layer GRU + CTC + HLG)

| Metric | Value |
|--------|-------|
| PER | **48.47%** |
| WER | **73.53%** |
| Spectrogram baseline WER | 100% |

---

## Decoding stack

1. CTC phoneme posteriors (beam 50, no LM for PER).
2. **HLG** = H (CTC topology) ∘ L (LibriSpeech lexicon FST) ∘ G (4-gram KenLM).
3. WER via Levenshtein on word sequences.

---

## OpenAlterEgo roadmap

Listed as Phase E / open-vocabulary in [`../../14-systematic-roadmap.md`](../../14-systematic-roadmap.md). Requires phoneme labels, CTC decoder, and SPD feature path.
