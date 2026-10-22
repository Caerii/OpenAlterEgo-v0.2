# EMG-UKA session-eight result: selection misses its primary control

The [method and selection](73-emg-uka-session-eight-selection-lock.md)
were committed and pushed at `bec2157a72de511f5833956446501853a2724073`
before the session-008 waveforms were scored. The frozen
[evaluation record](results/emg-uka-session-008-evaluation-v1.json) checks
the selection record hash and records the holdout content digest.

| Program | Zero-based recorded channels | Session-008 prompts correct |
| --- | --- | ---: |
| Seven-session selected three | (1, 3, 5) | **6/10** |
| Fixed three | (0, 1, 2) | **6/10** |
| Seed-42 random three | (0, 1, 3) | **5/10** |
| All six, higher channel cost | (0, 1, 2, 3, 4, 5) | **7/10** |

The primary selected-versus-fixed difference is **zero**. On paired
prompts the selected program gets one right that fixed misses and gets
one wrong that fixed gets right. It exceeds the random control by one
prompt but falls one behind the costlier six-channel reference. Its
leave-one-session-out development mean was 0.83; session-008 accuracy
is 0.60. With only ten prompts, these counts do not support a stable
generalization estimate. All conditions classify the same ten repeated
*audible* sentences, not continuous silent speech.

After exposing the result, we scored all 20 equal-cost three-channel
programs on session 008 as a **post hoc diagnostic**. The
[audit record](results/emg-uka-session-008-rank-audit-v1.json) retains
every development and session-008 score. Development and session-008
program ranks have Spearman correlation **0.690**, but the frozen choice
ranks fifth on session 008. The hindsight best (1, 4, 5) gets **8/10**;
the frozen choice has **2/10 prompt decision regret**. The hindsight
choice is not a prospective result. Its development mean was 0.80,
only 0.029 below the selected program's 0.829 across seven folds.

This is a useful failure mode for the search harness: a moderately
correlated whole-program ranking does not guarantee the top decision.
The two leading programs differ by one channel, so exhaustive search
or a local swap on the same development score would still prefer the
frozen loser. We need uncertainty-aware selection and more independent
sessions, not just faster optimization of that score. The real signals
show session variation of an existing montage; they do not predict what
moving an electrode, changing its contact impedance, or manufacturing
OpenAlterEgo would do.
