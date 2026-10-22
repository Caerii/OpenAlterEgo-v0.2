# Frozen seven-person differential-program method result

**Outcome: the declared success rule failed.** The
[protocol](44-differential-program-method-holdout-protocol.md) and scoring
code were pushed in `a5829be`. Day-1 differential caches for subjects
06–08 and 17–20 were then built from publisher-verified Hyser records.
The [selection record](results/hyser-differential-holdout-v1-selection.json)
was pushed in `e279673` before any of these subjects' day-2 **differential**
caches were created. Its canonical SHA-256 is
`5c4d058155bb3c9420bc8f6f4b456563ae8db1ea5f017d6ed03c1b5dadedd2c6`.
The [score record](results/hyser-differential-holdout-v1-score.json) binds
that hash to exact source identities, code and pair maps. The subjects'
day-2 raw records and monopolar features had been used in other project
experiments, so this is a **method-specific** evaluation, not an untouched
participant holdout.

The selected program is the maximum day-1 three-fold ridge-CV result among
5,000 feasible eight-pair/12-contact programs. The same day-1 ridge decoder
is evaluated on day 2 for the selected program, the fixed center-wedge
differential program, and the earlier fixed eight-site monopolar reference.
The outcome is balanced accuracy over recorded forearm gestures.

| Subject | Selected differential | Fixed differential | Paired gain | Fixed eight-site recording |
| --- | ---: | ---: | ---: | ---: |
| 06 | 0.211 | 0.461 | −0.250 | 0.583 |
| 07 | 0.495 | 0.451 | +0.044 | 0.539 |
| 08 | 0.245 | 0.186 | +0.059 | 0.482 |
| 17 | 0.279 | 0.191 | +0.088 | 0.319 |
| 18 | 0.453 | 0.358 | +0.095 | 0.461 |
| 19 | 0.682 | 0.621 | +0.061 | 0.697 |
| 20 | 0.167 | 0.127 | +0.039 | 0.176 |
| **Mean** | **0.362** | **0.342** | **+0.019** | **0.465** |

The rule improved six of seven people over the fixed differential program,
but its −0.250 loss on subject 06 violated the no-large-regression gate.
The mean gain missed the +0.05 gate, and selected differential accuracy was
below the fixed eight-site reference for **all seven** people. The
predeclared rule therefore cannot be recommended as a hardware selector.
High day-1 CV values (0.912–0.966) did not guarantee day-2 transfer.

The fixed differential wedge is illustrative and weak for some people, so
its six pairwise wins alone would have overstated the value of search. The
eight-site recorded-channel reference uses a shared reference contact and
a different signal path; it is a practical challenge comparator rather
than a controlled analog-circuit experiment. Digitally subtracting
publisher-preprocessed signals does not recreate contact impedance, analog
common-mode rejection, AFE noise or physical reseating. Hyser is forearm
gesture data, not facial silent speech.

The measured-pair harness remains useful for falsifying search rules and
studying intervention rankings. The next model should target day-to-day
pairwise **decision stability** and emit uncertainty or abstention, while
preserving the fixed recorded-site benchmark. A new facial-array dataset
and physical intervention would still be needed before claiming a wearable
advantage.
