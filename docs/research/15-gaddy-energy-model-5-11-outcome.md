# Locked 5-11 energy-model outcome

The [protocol and implementation](14-gaddy-energy-model-5-11-lock.md) were
committed and pushed at `be3b95b6adb7062eaa74be0a88e61d528d3a7120`
before session 5-11 was extracted. The downloaded publisher archive passed
its published MD5. Of the first 100 matched trials, 71 had eligible
`sentence_index` values. No 5-11 calibration, fitting or model selection was
performed. There were zero matching `(book, sentence_index)` keys against
the 382 distinct eligible training keys. [The result file](results/gaddy-energy-model-5-11-locked.json)
contains the full metrics and paired utterance-bootstrap intervals.

| Vocalized 5-11 metric | Current ridge | Global activity | Full causal-history model | Control |
| --- | ---: | ---: | ---: | ---: |
| All-frame log10-RMS MAE | 0.610 | 0.432 | **0.380** | Training mean 0.848 |
| Oracle voiced-interior log10-RMS MAE | 0.664 | 0.447 | **0.446** | Training mean 0.914 |
| Voiced-interior 50 ms change MAE | 0.277 | 0.219 | **0.219** | **Zero change 0.195** |

The primary **energy-engineering decision passed**. The paired
utterance-level advantage of full-model versus current ridge was 0.248
log10-RMS MAE, 95% bootstrap interval [0.229, 0.271]. Versus the
global-activity model it was 0.043, interval [0.032, 0.053]. A one-second
past feature shift raised the full model's all-frame MAE to 0.956, with
control-minus-aligned advantage 0.580, interval [0.554, 0.606]. These are
conditional estimates over 71 utterances from one speaker and one session,
not population-level confidence intervals.

The separately locked **expression decision failed**. The model's changes
between adjacent continuously voiced frames were less accurate than a
zero-change prediction. Within the voiced interior, the full model's
absolute energy error also barely differed from the global-activity model.
Its strong all-frame MAE improvement therefore should not be presented as
faithful stress, pitch, emotion or voice reconstruction. The oracle voiced
mask depends on recorded audio and the pitch heuristic can err. All trials
here contain actual phonation, which can affect both physiological activity
and electrode pickup. These data cannot establish performance for silent
speech.

The next modeling priority is a lower-frequency, within-speech contour
target and an explicit speech-activity component, followed by paired
silent/voiced evaluation. Those choices need a fresh untouched test split or
session. Session 5-11 is now development evidence and cannot be reused as an
unseen confirmation set.
