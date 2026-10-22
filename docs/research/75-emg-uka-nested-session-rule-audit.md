# Session-robust rules do not beat a random subset on EMG-UKA

The [frozen session-008 outcome](74-emg-uka-session-eight-outcome.md)
showed that optimizing mean development accuracy picked an unstable top
program. We therefore ran a **retrospective** nested-session comparison
of four selection rules on all eight speaker-008 audible sessions. The
[runner](../../software/python/scripts/emg_uka_cross_session_rules.py)
uses the same checked corpus import, five-bin EMG features, ten repeated
prompts, ridge classifier, and all 20 three-of-six recorded-channel
programs. For each outer target, the rule sees only the other seven
sessions: six fit the model and the seventh supplies one inner score,
rotated seven times. After program selection, seven sessions fit the final
model and the excluded target provides ten prompt outcomes. The target-008
mean-rule choice is checked against the [previous frozen selection](results/emg-uka-session-008-selection-v1.json).

The four rules are inner mean; mean minus one population standard
deviation; mean of the worst two folds; and minimum fold. They all use
the same 140 inner program fits per target. Fixed three, a seed-42 random
three, and all six recorded channels provide controls. The complete
[result](results/emg-uka-cross-session-rules-v1.json) includes source
content digests, every inner fold score, every target choice, every
program's outer outcome, and hindsight oracles.

| Outer session | Mean | Mean − SD | Fixed three | Random three | All six | Best three in hindsight |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 001 | 7 | 8 | 6 | 7 | 8 | 10 |
| 002 | 7 | 7 | 7 | 7 | 7 | 9 |
| 003 | 9 | 9 | 7 | 9 | 10 | 10 |
| 004 | 7 | 7 | 6 | 8 | 7 | 9 |
| 005 | 8 | 7 | 10 | 9 | 9 | 10 |
| 006 | 9 | 9 | 9 | 9 | 9 | 9 |
| 007 | 8 | 8 | 9 | 7 | 10 | 10 |
| 008 | 6 | 6 | 6 | 5 | 7 | 8 |
| **Total / 80** | **61** | **61** | **60** | **61** | **67** | — |

The worst-two and minimum rules also total **61/80**. Mean selection
beats fixed on three sessions, loses on two, and ties on three. Against
seed-42 random, it wins two, loses two, and ties four. Mean and
mean-minus-SD each have an average **1.75-prompt hindsight regret per
session** to the best equal-cost subset, computed only after seeing the
target. All six channels win or tie against the mean-selected three in
every session, at twice the recorded-channel cost.

This does **not** establish that random search is equally good in general;
the comparator is one fixed random draw and there are only eight
correlated sessions of one person with ten repeated audible prompts.
It does show that these simple session-risk objectives do not deliver
the sought advantage in this corpus. The 20-program search space is
fully scored, so optimization speed or missed combinations cannot explain
the outcome. The bottleneck is predicting which program transports to a
new session. All eight sessions, including 008, are now exposed; using
this table to tune another rule would be development work and needs a
different independent corpus or future recording for validation. Virtual
channel subsets also leave electrode relocation and actual AFE behavior
untested.
