# Locked Gaddy third-session outcome

The [protocol](07-prosody-third-session-lock.md) was pushed at OpenAlterEgo
commit `f996d71532274aa8d5eafe7be4fba4e4bc3b5db7` before session 5-8 was
extracted. The archive passed its published MD5 check. The importer found 67
non-reference test utterances among IDs 0–99; training used 83 non-reference
utterances from session 5-5. No 5-8 acoustic target was used to choose the
model or normalize its EMG. The normalization used all **unlabeled** 5-8 EMG
feature frames, so it is an offline transductive check.

| 5-8 target | Raw EMG MAE | Normalized EMG MAE | Mean prior MAE | Decision |
| --- | ---: | ---: | ---: | --- |
| Voiced-frame log audio RMS (primary) | 10.445 | **0.720** | 0.837 | Passed predefined threshold |
| All-frame log audio RMS | 12.763 | 0.756 | 0.849 | Secondary improvement |
| Voiced-frame F0, Hz | 740.45 | 80.35 | **73.68** | Failed secondary comparison |

The primary MAE reduction is approximately 13.9% relative to the constant
prior. Shuffling training EMG yielded 0.831 voiced-frame log-energy MAE in
the raw-feature control; a normalized shuffled control was not run. The
[full result](results/gaddy-prosody-locked-5-8.json) records trial IDs, frame
counts, R², and all controls. The raw model again extrapolated badly across
session gain. Session 5-8 does not validate the earlier 5-6 exploratory
number; it validates only the single rule locked before this extraction.

This result supports further testing of a limited acoustic-energy cue in
voiced speech. It does not recover a silent utterance's intended pitch,
nuanced emotional state, speaker identity, or speech with zero articulatory
movement. Repeated prompts, one speaker, crude F0 targets, and access to the
whole unlabeled test session limit the inference. The next prospective check
should ask whether a *causal* calibration available at utterance time retains
the energy advantage and whether it survives unseen text or another corpus.

**Later metadata correction (after this outcome):** An audit of the first-100
non-reference utterances found no overlapping book/sentence keys or identical
text strings between sessions 5-5 and 5-8. The repeated-prompt caveat above
was conservative and does not describe these actual subsets. The one-speaker,
alignment, full-session normalization and acoustic-target limitations remain.
