# Past channel drift is a weak proxy for next-day drift

The [additive SilentWear search](89-silentwear-additive-transfer.md) finds
higher two-day selection scores without improving next-day command accuracy.
Before adding a penalty for supposedly unstable channels, we measured
whether that instability is itself repeatable and predictive.

The [diagnostic](../../software/python/scripts/silentwear_drift_transport.py)
uses the same public 14-channel event features for S01–S03. For each of the
nine command/rest classes it computes a channel's two-feature centroid,
then removes the channelwise mean over classes. This retains the
**class-specific pattern** while excluding a shared recording-wide
amplitude offset. The mean squared difference between day-1 and day-2
patterns is early drift. The distance from their average to day 3 is
future drift. Between-class pattern energy is a simple signal-strength
diagnostic. Cross-day quantities use the first two days' within-class
variance with a 0.01 floor. Both raw and recording-robust features are
examined.

We also estimate the day-1-to-day-2 drift twice: once using each day's
first two batches and again using its last three. This checks whether
the channel ranking is even reproducible within that *same pair of days*.
All rank correlations are Spearman over the **14 dependent channels**;
they are descriptive, not participant-level inference.

| Person | Normalization | Same day-pair drift replication | Early drift → day-3 drift | Early signal → day-3 signal |
| --- | --- | ---: | ---: | ---: |
| S01 | Raw | 0.684 | 0.187 | 0.930 |
| S01 | Recording robust | 0.565 | 0.174 | 0.965 |
| S02 | Raw | 0.521 | 0.112 | 0.789 |
| S02 | Recording robust | 0.358 | 0.446 | 0.938 |
| S03 | Raw | 0.908 | 0.345 | 0.798 |
| S03 | Recording robust | 0.745 | 0.385 | 0.903 |

The six descriptive correlations average **0.630** for repeating the
same day-pair drift estimate across batches, but only **0.275** for
predicting future drift. Early signal strength ranks next-day signal
strength much more consistently, averaging **0.887**. In other words,
the early drift metric measures a real change between the first two
recording days, yet that change is a poor estimate of which channels
will change most on the next day in this small sample. Simply penalizing
past drift as an intrinsic channel property lacks support here.

The [record](results/silentwear-channel-drift-transport-dev03-v1.json)
contains all 14 channel metrics per person and normalization, source
manifest hashes, batch provenance, and the correlations. All day-3
recordings were exposed in earlier work, so this is a retrospective
diagnostic, not a new holdout. Signal-strength rank consistency also
does not imply a six-channel program will recognize commands well:
channels can be redundant, class relationships can change, and the
event features omit richer articulation timing. These observations
cannot identify electrode motion, contact impedance, or anatomy as
the cause of any measured shift.

The next search model should be judged by its **future-session program
decisions**, not by a plausible drift penalty or a higher early score.
The stable class-signal ranking is a candidate input to test, with
controls for channel redundancy and an independent facial-session gate.
