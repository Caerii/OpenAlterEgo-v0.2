# Extra calibration mostly helps decoder adaptation

The [rank-transport audit](80-mixed-program-rank-transport.md) found that
later-day query repetitions rank mixed programs more consistently than the
initial two-support selector. A query-A-selected program did better on
disjoint query B, but that probe gave it extra labels without matching the
final decoder's calibration. We ran the [matched comparison](../../software/python/scripts/hyser_four_support_selection.py)
to separate **program selection** from **decoder refitting**.

For each of the same 12 previously exposed Hyser forearm participants, the
runner draws the same seed-42 pool of 5,000 complete eight-signal programs.
It uses the original two labeled day-two repetitions per gesture to choose
the first program. After fitting on those repetitions, it uses a disjoint
additional set A to choose a second program. The first choice, second
choice, and fixed eight-site layout are then all **refit with identical day-one
data plus the original and A support trials** (total support weight four per
class), and scored only on the remaining B trials. Thus all three final
decoders have the same labeled training budget. The [result record](results/hyser-four-support-selection-12-v1.json)
retains source hashes, split indices, choices, contacts, and outcomes.

| Selection, all refit with four-support data | Mean B balanced accuracy |
| --- | ---: |
| Original two-support choice | 0.865 |
| Extra-A-selected choice | **0.879** |
| Fixed eight-site layout | 0.830 |

The extra-A selector beats the original selector by **+0.0137** paired mean,
with 9/12 wins but three substantial losses and a 10,000-draw participant
bootstrap interval **[−0.0257, +0.0505]**. It beats the matched fixed layout
by **+0.0496** paired mean, with interval **[+0.0176, +0.0823]** and 10/12
wins. The original two-support choice also beats the fixed layout on this
comparison by about +0.036. The extra labels therefore do not establish a
reliable **incremental selection** improvement.

In the preceding audit, the original choice scored 0.800 on B when its
decoder used only the first two support repetitions. With the same choice
but four-support refitting here, B rises to **0.865**. This is a comparison
within the same participant splits and selected programs, and it points to
decoder adaptation as the larger measured effect. Participant 06 has one
remaining trial for one gesture that cannot enter a disjoint A/B partition;
its index is recorded and omitted from B in both comparisons.

The A/B partition is within **one recording day**. All 12 participants have
been examined repeatedly, and this follow-up was motivated by their query
rankings. These numbers are retrospective development evidence, not a locked
new-session validation or a causal electrode-intervention result. The Hyser
signals are forearm measurements with digital channel differences; they do
not measure facial silent speech or an analog differential wearable.

The next program-search gain needs a selector whose advantage survives a
*new day or person* at a fixed calibration budget. Until then, the harness
should account for the value of collecting and fitting a few more labeled
examples before spending more compute on optimization.
