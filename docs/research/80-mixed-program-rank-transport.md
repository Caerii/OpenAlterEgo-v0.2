# The mixed-program bottleneck is transporting rankings across calibration

The [guided search result](78-guided-compositional-search-generalization.md)
showed that finding a much higher support score did little for disjoint
later-day performance. We directly measured candidate-rank transport for the
new eight-signal mixed grammar rather than running another optimizer.

The [diagnostic runner](../../software/python/scripts/hyser_mixed_rank_transport.py)
replays the same seed-42 pool of **5,000 complete programs** for each of 12
previously exposed [Hyser](https://physionet.org/content/hd-semg/2.0.0/)
forearm participants. Selection uses half day-one three-fold CV and one
quarter each from two separate labeled day-two support repetitions. The
decoder is trained on day one plus both support repetitions. Remaining
day-two trials are alternated within each gesture into disjoint query A and
query B. Candidate scores on either query partition are **diagnostics only**;
no query result enters the support selection.

One participant (06) has just one remaining trial for one gesture. That
trial is omitted from the A/B rank comparison, recorded by index, and
retained in the full-query accuracy. The [result record](results/hyser-mixed-rank-transport-12-v1.json)
retains recording and candidate-pool hashes, all split indices, selections,
rank statistics and outcomes. The selected full-query score reproduces the
[earlier two-support result](77-two-support-high-dimensional-program-search.md)
exactly for all 12 people.

| Across the 12 people | Mean |
| --- | ---: |
| Spearman: support selection vs query A | 0.394 |
| Spearman: support selection vs query B | 0.398 |
| Spearman: query A vs query B | **0.706** |
| Top-decile overlap: support vs query A / B | 0.259 / 0.263 |
| Top-decile overlap: query A vs query B | **0.457** |
| Selected program, full-query balanced accuracy | 0.821 |
| Median of 5,000 candidates, full-query accuracy | 0.739 |
| Best of 5,000 **using full-query labels** | 0.912 |

The query-A-to-query-B rank correlation exceeds the support-to-query-B
correlation by **+0.308** on average; a 10,000-draw participant-bootstrap
interval is **[+0.276, +0.340]**. Those 5,000 related programs are not
independent replicates, and all participants have been inspected before.
The interval describes participant variation in this retrospective replay,
not out-of-sample assurance.

As a limited same-day information probe, choosing the best candidate on
query A yields **0.857** mean accuracy on disjoint query B, versus **0.800**
for the original support-selected candidate on B. The paired difference is
**+0.057**, with participant-bootstrap interval **[+0.019, +0.091]** and
10/12 wins. This selection consumes additional labeled query-A repetitions
and uses the same day as B; it is **not** an equal-calibration-cost comparison
or evidence of transfer to a new day. The full-query oracle of 0.912 uses
the evaluation labels themselves and is only a headroom diagnostic.

The result suggests that the grammar contains better measured-signal
programs, while the two-support objective has limited ability to identify
them. The next falsifiable test is a **matched four-support comparison**:
choose using the extra labeled A repetitions, refit both the chosen and fixed
programs with the same added labels, and score B. Then check whether the
gain survives across participants and, eventually, new sessions and facial
data. Forearm digital differences still cannot validate the analog front
end or facial electrode physics.
