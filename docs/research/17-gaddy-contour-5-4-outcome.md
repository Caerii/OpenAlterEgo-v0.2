# Locked 5-4 slow-contour outcome

The [protocol](16-gaddy-contour-5-4-lock.md) and the Python evaluation code
were committed and pushed at `f767d58d55f174976a56acbbc11f13c688a5bc40`
before any 5-4 trial was extracted. The publisher archive passed its MD5.
The first 100 matched trials contained 60 eligible utterances and 14,058
frames. There were zero overlapping `(book, sentence_index)` keys against
the 453 distinct eligible training keys. No 5-4 tuning or calibration was
performed. The [machine-readable result](results/gaddy-contour-activity-5-4-locked.json)
contains all metrics and intervals.

| 5-4 vocalized-data result | 200 ms slow-contour change MAE |
| --- | ---: |
| Zero change | 0.3394 |
| Global-activity speech head | 0.2989 |
| Full soft-gated head | **0.2724** |
| Dense latent soft-gated head | 0.2740 |
| Top-two sparse latent soft-gated head | 0.2721 |

The prespecified **slow-contour decision passed** on 6,925 uninterrupted
audio-defined activity frames. Across 60 utterances, the mean paired
control-minus-full advantage was 0.0660 versus zero change (95% bootstrap
interval [0.0538, 0.0786]) and 0.0234 versus the global-activity head
([0.0166, 0.0303]). The separate audio-defined activity classifier scored
0.979 average precision, 0.967 ROC AUC and 0.942 F1 at probability 0.5.

The prespecified **sparse-prior decision did not pass**. Although the
top-two version had a slightly lower frame-weighted MAE, its paired
utterance-level advantage over the full soft-gated model was 0.00194 with
interval [-0.00357, 0.00842]. The 75% zero code fraction verifies the
computational sparsity intervention, but does not identify motor units or
muscle synergies. Its eight nonnegative dictionary patterns are only a
candidate representation of observed electrode features.

These intervals resample utterances from one speaker and one session, not
people. The target, activity mask and energy changes come from actual
vocalized audio. EMG during phonation may include muscle and sensor effects
that do not persist during silent articulation. The result establishes
cross-session prediction of a slower vocalized energy contour under this
fixed montage; it cannot establish intended silent prosody, emotion,
original-voice reconstruction or causal electrode-layout effects. Session
5-4 is now opened evidence, not another unseen test.
