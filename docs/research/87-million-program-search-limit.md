# One million programs: exact search is feasible, selection is still limiting

We extended the [100,000-program audit](82-large-pool-scaling-and-facial-data.md)
to **one million complete eight-signal programs** for all 12 available,
already exposed Hyser people. Each program selects one adjacent contact
edge in each of four forearm grids and two signals from its three recorded
modes. The physical cost remains eight array contacts, one shared reference,
and eight acquisition channels. The approximately 12.7-billion-program
grammar is far larger than the sampled pool.

The [runner](../../software/python/scripts/hyser_candidate_scale_audit.py)
now samples the seed-42 prefix **once** and reuses it across subjects after
checking that their contact and pair topology matches. Budgets of 5,000,
20,000, 100,000 and one million are nested prefixes of that same pool.
The selection objective remains half day-one three-fold balanced accuracy
plus one quarter of each of two separately labeled day-two support
repetitions. The chosen program is fit with both support repetitions and
evaluated on disjoint remaining day-two queries. Rust scores each candidate
exactly with the same ridge proxy; no surrogate approximation is involved.

| Candidate budget | Mean later-day query balanced accuracy, 12 people |
| ---: | ---: |
| 5,000 | 0.8213 |
| 20,000 | 0.8061 |
| 100,000 | **0.8254** |
| 1,000,000 | 0.8175 |
| Fixed eight-site control | 0.7733 |

From 100,000 to one million candidates, the selection score improved by
**0.0215** on average, but future-day accuracy changed by **−0.0078**:
four wins, six losses, and two ties. A 10,000-draw person bootstrap of the
paired difference, seed 77, gives a descriptive 95% interval of
**[−0.0376, +0.0218]**. The largest loss was approximately −0.103
(subject 05), and the largest gain approximately +0.081 (subject 20).
There is no evidence here of a reliable gain from increasing the budget
tenfold beyond 100,000. The result does not prove a universally optimal
budget, since all budgets and subjects are already exposed.

The [full record](results/hyser-candidate-scale-1m-rust-12-v2.json) retains
the shared pool SHA-256, source hashes, support/query indices, each selected
program and physical contacts, selection score, query score, fixed control,
and timing. Previously reported 5,000-program outcomes matched all 12
subjects; every choice for the three subjects from a separate one-million
replay [matched exactly](results/hyser-candidate-scale-1m-rust-3-v1.json).
Sampling the shared million-program pool took
14.9 s once, and median exact scoring across the five selection folds took
34.7 s per subject in this local run. These are workload-specific timings,
not portable throughput guarantees. Reusing the pool saves roughly a
sampling pass for every subsequent subject.

The engineering implication is specific: search throughput is sufficient
to test million-scale candidate pools, but **the current selection proxy
cannot reliably rank the best future-day design at its extreme**. The next
algorithm should be evaluated on future-session decision regret and paired
gain under a fixed hardware budget, with uncertainty and an option to
abstain. More candidate generation by itself is unlikely to deliver the
desired advantage. Hyser's measured forearm digital differences also do
not validate facial silent speech, analog differential electronics,
unmeasured electrode placement, or device reseating physics.
