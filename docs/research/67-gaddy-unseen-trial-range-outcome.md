# Frozen Gaddy trial-range outcome

The [protocol](66-gaddy-unseen-trial-range-lock.md) was pushed at
`19a1661` before extracting trial IDs 100–199 from parallel sessions
5-4 and 5-8. The archive matched its published MD5. The
[scorer](../../software/python/scripts/gaddy_locked_trial_range.py) used the
frozen models, channels, first-ten-valid-utterance unlabeled calibration,
and frame-level log acoustic energy MAE. Its [result](results/gaddy-locked-trial-range-100-199-v1.json)
retains valid trial IDs, frame counts, source manifests, and paired
per-utterance errors. Raw recordings and feature caches stay outside Git.

| Query ID range | Multi-session six | Earlier-session six | Index-fixed six | All eight | Evaluated valid utterances |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5-4, 100–199 | **0.4843** | 0.4969 | 0.5096 | **0.4736** | 69 |
| 5-8, 100–199 | **0.4663** | 0.5343 | 0.6050 | 0.5219 | 76 |
| Unweighted session mean | **0.4753** | 0.5156 | 0.5573 | 0.4977 | — |

The multi-session choice `(0,1,2,3,4,6)` beat both frozen six-channel
comparators in **each** range, meeting the primary directional check.
Its mean MAE was 0.0821 lower than the index-fixed six. It also beat
all eight in the two-session mean, but **all eight was better on 5-4**.
Thus the result supports a useful sparse channel choice for this assay,
not universal dominance of a six-channel wearable.

The [paired utterance bootstrap](results/gaddy-locked-trial-range-100-199-audit-v1.json)
resamples utterances within each session with seed 42 and recomputes
frame-weighted session MAE. The 95% descriptive interval for selected
minus index-fixed six in the two-session mean is **−0.0920 to −0.0736**;
against all eight it is **−0.0275 to −0.0180**. These intervals describe
the recorded utterances conditional on just two already exposed sessions.
They cannot quantify speaker, recording-day, electrode-placement, or
silent-speech uncertainty.

This was a locked, previously unscored **trial-range** check, not a fresh
session holdout: the seven parallel sessions and some of their first
100 trials informed prior research. The endpoint is vocalized acoustic
energy from one speaker; the channel masks are virtual subsets of the
same eight recorded channels. The next decisive data should vary
session and speaker while permitting physically meaningful electrode
layouts and a silent-speech decoding endpoint.
