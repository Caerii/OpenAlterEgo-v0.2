# One-cell displacement as a measured program-search stress test

The support-rotation audit found that selected contacts were unstable.
This experiment uses simultaneous HD-sEMG array recordings to ask a
more physical question: would a proposed eight-contact program still
work if every contact on each grid were translated one grid cell between
day one and day two?

For each shared-reference channel, map its physical contact to the
neighbor. For each digital differential channel, map *both* physical
contacts and use the recorded adjacent pair at the new position. The
five displacements are zero, one row up/down, and one column left/right.
Only edge/mode grammar choices valid under all five shifts are allowed.
That leaves 180 options per grid, or **1,049,760,000** complete four-grid
programs. Each still uses eight array contacts, one common reference,
and eight AFE channels. Grid coordinates are topological; no millimetre
displacement is inferred.

Day-one data always comes from the nominal sites. Day-two labeled
support and disjoint query features come from the translated sites.
This models a session-to-session placement shift rather than translating
both sessions together. One support repetition per gesture is used for
weight-4 decoder adaptation. The dense recording supplies all shifted
positions simultaneously, so this is a retrospective design-screening
setting; an eight-contact wearable could not observe all five positions
during its own calibration.

## Development comparison

Subjects 01–05 use the same 5,000 seeded complete candidates from the
restricted grammar. For each candidate, calculate day-one CV and
day-one-to-support accuracy for each displacement. The nominal selector
maximizes their half-and-half score at zero shift. The robust selector
maximizes the *minimum* half-and-half score across five shifts. Evaluate
both on shifted day-two query trials, with the fixed eight-site layout
as a same-budget reference.

| Program | Nominal query | Mean over five shifts | Mean personwise worst shift |
| --- | ---: | ---: | ---: |
| Nominal selected | 0.765 | 0.745 | 0.698 |
| Worst-shift selected | 0.776 | 0.760 | 0.722 |
| Fixed eight-site | 0.704 | 0.707 | 0.659 |

Across these five development subjects, the robust selector improved
mean shifted accuracy by +0.015 over the nominal selector, and mean
personwise worst shift by +0.024. It exceeded the fixed layout by
+0.053 in mean shifted accuracy. Subject 02 and subject 04 still had
negative gain versus fixed. The [source artifact](results/hyser-contact-shift-dev05-v1.json)
records all five per-subject displacements and program choices; the
[replay code](../../software/python/scripts/hyser_contact_shift_search.py)
records a wall time of 7.3–9.5 seconds per person on this local run.
These timings are exploratory, not controlled benchmarks.

## Method-check protocol

Freeze the same 5,000-candidate seed, five shifts, and worst-shift
selection rule for subjects 06–08 and 17–20. Commit the choices before
computing their day-two query accuracies. Primary effect: robust minus
nominal selected program, mean balanced accuracy over five shifts per
person. Report per-person gains and mean personwise worst-shift accuracy.
The fixed eight-site program is the strong reference. A compelling
method-level gain would require robust minus nominal mean at least
+0.02, robust minus fixed mean at least +0.05, robust beating fixed
for five of seven people, and no robust-minus-fixed loss below -0.05.
These criteria are deliberately stronger than the current development
result; a failed check is useful evidence.

This is still forearm gesture data. The digital difference signals
omit electrode-skin and analog front-end behavior. Translations do not
model rotation, independent contact displacement, skin deformation,
or facial anatomy. Subjects and day-two recordings were exposed in
other Hyser work, so the method check cannot be treated as independent
participant validation.
