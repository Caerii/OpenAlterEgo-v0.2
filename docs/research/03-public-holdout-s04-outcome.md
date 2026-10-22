# SilentWear S04 locked outcome

The S04 method was committed and pushed as `d5626add27827edbb74939ca839a8dfdf82a2e59`
before its files were downloaded. The publisher's 15 S04 silent HDF5 files were
then downloaded at the pinned dataset revision and checked against their
publisher SHA-256 values. The fixed code evaluated session 1 for fitting,
session 2 for subset selection, and session 3 for reporting. No S04 result was
used to change the method before this run.

| S04 result | Frozen normalized rule | Raw-feature comparator |
| --- | ---: | ---: |
| Pairwise ranking accuracy | 0.70 | 0.50 |
| Selected subset | `alternate` | `mid_index` |
| Selected subset session-3 balanced accuracy | 0.2275 | 0.2301 |
| Best of five recorded subsets, session-3 balanced accuracy | 0.2635 | 0.2602 |
| Decision regret | **0.0360** | 0.0301 |

The preregistered ranking hypothesis (>0.5) passed. The regret hypothesis
(≤0.03) failed. The normalized selector ranked most pairs correctly but chose
`alternate` because its session-2 score (0.278) slightly exceeded `spread`
(0.268); on session 3, `alternate` reached 0.228 and `spread` 0.264. One
participant and five candidates provide no population-level assurance. The
complete per-candidate metrics and 15 input hashes are in
[the S04 result](results/silentwear-s04-locked-result.json).

This result is now a development observation. S04 cannot be reused as an
untouched person-level holdout for a revised selector. The next offline
hypothesis is that uncertainty across session-2 batches should influence the
choice: a candidate with a tiny mean advantage and high batch variability
may be less reliable than the runner-up. Evaluate that idea transparently on
S01–S04, then use a separate corpus or new untouched split for a fresh test.

These outcomes concern virtual subsets of SilentWear's existing recorded
channels. The feature-drift perturbation is uncalibrated and does not establish
performance for a relocated electrode or an OpenAlterEgo physical wearable.
