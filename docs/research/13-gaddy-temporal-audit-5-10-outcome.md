# Locked 5-10 temporal-audit outcome

The [protocol](12-gaddy-temporal-audit-5-10-lock.md) and implementation were
pushed at commit `0d4b316` before 5-10 extraction. The archive matched the
publisher MD5. IDs 0-99 contained 73 non-reference trials: the earliest ten
were used for unlabeled past-only calibration and the remaining **63** for
evaluation. There were zero overlapping book/sentence keys and zero matching
normalized text strings against the 5-5 training subset.

| Control minus aligned EMG, all-frame log10-RMS MAE | Mean paired utterance advantage | 95% utterance-bootstrap interval |
| --- | ---: | ---: |
| Training-mean prior | +0.263 | **[+0.248, +0.279]** |
| Shuffled-training EMG | +0.257 | **[+0.242, +0.272]** |
| Same-utterance EMG shifted by one second | +0.377 | **[+0.346, +0.412]** |

The lower endpoint is positive for all three controls, so the prespecified
primary decision **passes**. Bootstrap sampling used whole utterances, 5,000
draws and seed 42. At the frame-weighted level, calibrated EMG scored 0.551
MAE on 9,113 frames, against 0.794 for the mean prior and 0.788 for shuffled
training. The unnormalized forward-filtered model scored 0.541; calibration
was not necessary for the observed improvement. Secondary oracle-voiced-frame
F0 scored 56.18 Hz with calibration versus 68.14 Hz for the mean prior, but
the pitch target and voiced mask depend on recorded audio.

The [full result](results/gaddy-prosody-5-10-temporal-locked.json) records
every trial's error and the archive/session provenance. The intervals describe
variation across these utterances conditional on this one speaker, model and
session. They are **not** multi-speaker or multi-session uncertainty intervals.
The one-second circular shift demonstrates that the model uses temporal
alignment beyond a fixed utterance-level feature distribution; it does not
isolate fine prosody from onset/offset or phoneme timing.

This is stronger evidence for a vocalized EMG-to-acoustic-energy association
than the earlier frame-only checks. It remains unable to establish silent
intended loudness, pitch, affect or personal-voice reconstruction. Actual
phonation can affect both muscle activation and sensor pickup. A next gate
should condition on speech activity and content timing, then test residual
within-speech energy or stress in silent and voiced paired recordings. The
next numbered unused voiced parallel session 5-11 should remain reserved until that
model and decision rule are fixed.
