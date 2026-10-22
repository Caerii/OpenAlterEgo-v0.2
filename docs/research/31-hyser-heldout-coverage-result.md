# Hyser held-out coverage search: failed transfer gate

**Status:** completed, frozen method check. It uses [Hyser](https://physionet.org/content/hd-semg/2.0.0/)
forearm gestures to test recorded-site selection across days. It says nothing
directly about silent facial speech or an unbuilt wearable.

The [protocol](30-hyser-heldout-coverage-protocol.md) and code were pushed
at `9ffc285`; the exact [subject 04–05 selections](results/hyser-holdout-v1-selection.json)
were pushed at `7075912` before either subject's session 2 was downloaded.
The selection record has SHA-256
`2020fa6e26c2f01655128cfe5e2b3adf1ec0491451dbe697ef12041dff09d223`.
The [score record](results/hyser-holdout-v1-score.json) links to that hash and
to publisher-derived source hashes. Session 2 has 204 trials and all 34
nominal classes for both subjects. Subject 05 session 1 has 195 trials and
omits class 34; subject 04 session 1 has 202 trials and all 34 classes.

| Subject | Coverage day-1 CV | Fixed day-1 CV | Coverage day 2 | Fixed day 2 | Paired difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| 04 | 0.868 | 0.809 | **0.691** | 0.598 | +0.093 |
| 05 | 0.778 | 0.652 | 0.093 | **0.515** | −0.422 |
| Mean | 0.823 | 0.731 | 0.392 | **0.556** | **−0.164** |

The preregistered primary comparison fails. The greedy coverage rule gave
the better day-1 CV score for both people, yet catastrophically failed on
subject 05's new day. The one absent training class limits attainable
balanced accuracy by about 2.9 points if counted in the day-2 average; it
cannot explain a 42-point paired deficit. With two subjects, no population
inference is warranted. The result does show that same-day discriminability
and equal grid representation are inadequate selection objectives here.

Secondary day-2 results reinforce that conclusion. Unconstrained greedy was
0.652/0.088 for subjects 04/05; budgeted random was 0.549/0.299 without the
coverage cap and 0.598/0.034 with it. The naive all-256 ridge reference was
0.074/0.039 and should not be interpreted as a limit on array information:
it severely overfits a few hundred training trials. None of these secondary
comparisons changes the primary outcome.

## Next research decision

More evaluations of the present CV objective or faster ridge kernels will
find day-1 specialists more efficiently. The next candidate should estimate
which sites preserve *class relationships* when contact, amplitude, and
session conditions vary. Plausible simulations include per-site gain and
offset, correlated site dropout, spatial displacement across neighboring
recorded contacts, and class-independent covariance shift. Their ranges
should be calibrated using the observed between-day shifts on development
subjects 01–05, then fixed before testing a new cohort. Compare this robust
objective against the existing fixed layout, budget-matched random search,
and plain CV greedy. Calibrate predictions against per-subject day-2 effects,
not waveform appearance. This is a proposal, not a verified improvement.

The day-2 files remain outside Git. The acquisition script was corrected
after selection to preserve distinct filenames sharing a SHA-256 digest and
to validate paths after parent creation under concurrent Windows downloads.
That change did not alter the frozen feature, selection, or scoring code.

## Post-gate drift diagnostic

After seeing the held-out labels and scores, we explored unlabeled session-2
normalization. This is **post hoc** and transductive: each normalization used
the entire future session's feature distribution, so these numbers are not
prospective performance estimates and are unavailable at the start of a live
session. The same frozen channels and ridge fit were reused.

| Subject / layout | Raw | Target feature mean centering | Target feature mean and SD alignment |
| --- | ---: | ---: | ---: |
| 04 / coverage greedy | 0.691 | 0.721 | 0.706 |
| 04 / fixed grid | 0.598 | 0.711 | 0.721 |
| 05 / coverage greedy | 0.093 | 0.216 | 0.373 |
| 05 / fixed grid | 0.515 | 0.529 | 0.505 |

The subject-05 coverage collapse is partly compatible with a feature-scale
shift: per-feature mean and SD alignment raises it from 0.093 to 0.373. It
still trails the fixed layout. This diagnostic cannot identify the physical
cause or establish a deployable adaptation method. A future test should use
only an explicitly bounded, earlier *unlabeled* calibration segment, then
score subsequent trials under a frozen rule on new subjects.
