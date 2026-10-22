# A donor evidence gate abstains on all 12 exposed people

The [donor calibration experiment](84-cross-person-donor-calibration.md)
improved broad candidate ranking but did not reliably improve the final
eight-signal program. We added a **nested, person-level decision gate** to
ask a narrower question: does evidence from the other people justify
replacing the target's existing two-support choice with either donor policy?

For each outer target, the [gate](../../software/python/openalterego/research/donor_calibration.py)
uses only the other 11 people's query outcomes. Each of those 11 becomes an
inner pseudo-target in turn. The remaining 10 fit the donor-calibrated model
and supply the donor-only mean. The inner pseudo-target's query outcome
measures the gain or loss of each policy against its own two-support choice.
We resample the 11 **person-level gains**, with seed 77 and 2,000 draws.
A policy can replace the baseline only when the fifth percentile of its
bootstrap mean gain is strictly positive. If both qualify, the larger inner
mean gain wins; otherwise the gate abstains. This selection rule was set
before reading the resulting outer outcomes in this run. A unit test changes
an outer target's query outcomes to NaNs and confirms an identical gate
decision and evidence record.

| Policy | Outer targets passing gate | Outer query mean balanced accuracy |
| --- | ---: | ---: |
| Two-support baseline | 12 | 0.821 |
| Nested gate | 0 replacements; 12 abstentions | 0.821 |
| Donor-only, ungated diagnostic | — | 0.825 |
| Donor-calibrated, ungated diagnostic | — | 0.803 |

The donor-only inner mean gain ranges from roughly −0.07 to +0.01 across
outer exclusions, but its bootstrap fifth percentile is negative for
**every** outer target. The learned donor policy is likewise negative on
every outer exclusion. All 12 selected baseline candidate indices exactly
match the [previous artifact](results/hyser-donor-calibrated-12-v1.json).
The new [record](results/hyser-nested-donor-gate-12-v1.json) includes the
candidate-pool hash, source hashes, support/query splits, selected physical
contacts, each inner policy's mean gain/win/loss count/lower bound, and the
outer decisions.

Abstention does not prove that the two-support program is optimal. The
bootstrap treats person-level gains as resampling units even though their
models share donors, so its percentile is a descriptive decision threshold,
not a calibrated confidence guarantee. More importantly, the 12 people and
their query outcomes were already exposed while designing this research
sequence. The honest conclusion is that **this cohort provides no reason to
deploy the donor selector by default**. A frozen gate needs new people or
sessions for a prospective estimate.

This corpus measures forearm gestures from publisher-preprocessed monopolar
and digitally differenced signals. It cannot validate facial silent speech,
analog differential sensing, physical contact impedance, or an actual
wearable intervention. The next useful freely available data is a facial
multichannel array corpus with repeated sessions, allowing the same program
search and abstention protocol to be tested closer to the target anatomy.
