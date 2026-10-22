# Hyser design-ranking audit: optimize prediction before search scale

**Status:** exploratory development on subjects 01–05, whose day-2 outcomes
had already been seen. All results concern recorded forearm grid channels in
the [Hyser corpus](https://physionet.org/content/hd-semg/2.0.0/), not silent
facial speech or measurements at unrecorded positions.

The prior [held-out gate](31-hyser-heldout-coverage-result.md) showed that
plain day-1 CV could select a layout that failed badly the next day. We then
tested whether two physically motivated proxies improved the decision:

- **Dark-contact veto:** exclude a site if its median day-1 log RMS is more
  than 0.5 below the median of its immediate same-grid neighbors.
- **One-cell translation:** score each candidate on day-1 CV at its nominal
  positions and after shifting all eight sites one grid cell up, down, left,
  and right. Only interior sites are eligible, so each shift remains recorded.

The veto identified the two subject-05 channels with extreme day-2 drift
(sites 56 and 213). Yet [screened coverage](results/hyser-site-health-dev-v1.json)
reached only 0.147 next-day balanced accuracy for that subject, compared
with 0.093 before screening and 0.515 for the fixed layout. It helped subject
03 from 0.030 to 0.096, still far below fixed-grid 0.441. This is a useful
quality screen, not a sufficient transfer objective.

The [translation pilot](results/hyser-spatial-dev-v1.json) evaluated 1,000
seeded feasible eight-site candidates per subject and selected the one with
the highest *minimum* of its five day-1 CV scores. It produced mixed next-day
results: 0.429, 0.113, 0.389, 0.044, and 0.392 for subjects 01–05, versus
0.571, 0.242, 0.441, 0.598, and 0.515 for the fixed layout. The worst
subject 04 result makes this rule unsuitable for freezing as an improvement.
The grid step is topological; Hyser does not establish that it corresponds to
a particular millimetre displacement of a future wearable.

## Ranking, not only chosen-design outcomes

We implemented a batched day-pair ridge evaluator whose scores match the
single-layout reference, including a target class absent from training. It
lets us compare day-1 proxies to actual day-2 outcomes for the *same 1,000
candidate layouts* per subject. The [ranking audit](results/hyser-ranking-audit-dev-v1.json)
contains exact selections, scores, and publisher-derived source hashes.

| Subject | Spearman: nominal CV vs day 2 | Spearman: minimum translated CV vs day 2 | Spearman: site darkness vs day 2 | Fixed layout day 2 | Best of 1,000 on day 2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 01 | +0.143 | +0.166 | −0.103 | 0.571 | 0.717 |
| 02 | +0.150 | +0.081 | +0.091 | 0.242 | 0.372 |
| 03 | −0.266 | −0.106 | −0.047 | 0.441 | 0.631 |
| 04 | +0.064 | +0.150 | −0.666 | 0.598 | 0.696 |
| 05 | −0.159 | +0.154 | −0.478 | 0.515 | 0.515 |

The final column is an **oracle over day-2 labels**, a retrospective
headroom diagnostic, not a selectable performance estimate. Most subjects
have layouts better than fixed within the sampled pool, but the present
day-1 proxies cannot identify them reliably. Subject 05's sampled oracle
only matches fixed.

A [leave-one-person-out donor prior](results/hyser-donor-prior-dev-v1.json)
averaged day-2 outcomes for each candidate across the other four subjects,
then chose the best donor layout for the excluded person. It scored below
the fixed layout for all five people. Between-person transfer of a single
preferred layout is weak under this small pilot.

## Research decision

The bottleneck is **predicting the consequence of a design change**, not
evaluating more designs. The next model should learn from many *paired-day*
arrays, separate device/contact stability from gesture discrimination, and
output a calibrated prediction of next-day performance or regret for each
candidate. More subjects are needed before fitting a flexible surrogate;
hold out people and their future sessions from model choice. A fixed layout
remains the mandatory baseline, and any personalized search should be able
to abstain and choose it when evidence for improvement is weak.

No model here establishes physical causality. The next recordings are
already available publicly; a separate physical study would eventually be
needed to test new electrode placements and actual reseating interventions.
