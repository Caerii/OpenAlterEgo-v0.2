# Fixed mixed measurement with calibration: method check

The [five-person development result](49-calibration-conditioned-hardware-development.md)
suggests a fixed mixed measurement program can outperform an equal-cost
fixed shared-reference program **after** one labeled new-session trial per
gesture is used for calibration. Its weight-4 rule beat the comparator for
all five development people, with a +0.054 mean paired gain. We therefore
freeze weight 4 as the primary rule for subjects 06–08 and 17–20.

Both hardware programs use eight array contacts, one shared reference,
and eight AFE channels. The mixed program takes one shared-reference
potential and one adjacent digital difference sharing that contact in
each of four grids. The comparator is the established fixed eight-site
shared-reference placement. A third comparator uses all eight shared-reference
potentials from the **exact same contacts** as the mixed program.

For each person, train a ridge decoder on day 1 with alpha 10. The first
recorded day-2 repetition of every present gesture is labeled support.
Give each support class four times the per-class mass of day-1 data,
refit, and score the remaining day-2 repetitions. All programs receive
identical support and query trials. Weight 1 and zero support on the
same queries are secondary descriptions.

The primary outcome is per-person fixed-mixed minus established-fixed
balanced accuracy at support weight 4. A useful method-level result
requires a mean gain of at least +0.05, improvement for at least five of
seven people, and no loss worse than −0.05. Report every subject and the
same-contact comparison even if this rule fails.

The [runner](../../software/python/scripts/hyser_fixed_mixed_calibration_holdout.py)
writes an exclusive source-bound selection record before this specific
comparison is scored. These participants and day-2 recordings have been
used for earlier Hyser work, and the class-block support/query split is
not globally chronological. This is therefore a **method-specific replay**,
not independent participant validation or online wearable calibration.
Hyser measures forearm gestures; digitally subtracting its preprocessed
signals does not establish analog facial hardware performance.
