# Guided program search mostly improves its own selection score

The [two-support search](77-two-support-high-dimensional-program-search.md)
sampled 5,000 complete eight-signal programs from a grammar of about 12.7
billion. We tested whether a search that reuses promising physical program
parts extracts a better later-day decision from the **same evaluation budget**.

The [guided search](../../software/python/openalterego/research/guided_program_search.py)
starts with 500 random complete programs, then fills 500-program batches with
single-part mutations of the 32 current elites, recombinations, and new
random exploration. It scores only complete, feasible programs. The
[replay](../../software/python/scripts/hyser_guided_two_support_search.py)
compares that search with 5,000 unique uniformly sampled programs, randomly
ordered for the nested budgets. Both use the same selection function: half
day-one three-fold accuracy, one quarter each from two **separate** labeled
day-two support repetitions per gesture. Both fit the final decoder with the
same two support examples and evaluate the same disjoint day-two queries.
No query value enters either search.

The [result record](results/hyser-guided-two-support-12-v1.json) covers the
same 12 **previously exposed** Hyser forearm participants as the preceding
experiment. It retains source hashes, support and query indices, selected
programs and physical contacts, selection scores, query outcomes, and local
search timings. These are publisher-preprocessed potentials and digitally
differenced signals; they do not recreate an analog differential front end.

| Complete programs evaluated per person | Guided query balanced accuracy | Uniform query balanced accuracy | Guided minus uniform | Guided selection score | Uniform selection score |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 0.798 | 0.796 | +0.002 | 0.711 | 0.712 |
| 1,000 | 0.810 | 0.797 | +0.013 | 0.737 | 0.722 |
| 3,000 | 0.819 | 0.819 | +0.000 | 0.803 | 0.742 |
| 5,000 | **0.828** | 0.821 | **+0.007** | **0.819** | 0.743 |

At 5,000 evaluations, guided search wins for eight people, loses for three,
and ties for one. Its mean paired query gain is **+0.0066** with a 10,000-draw
participant-bootstrap interval **[−0.0205, +0.0296]**. Two of the three
losses are larger than 0.06. The average selected score rises by **+0.076**,
roughly an order of magnitude more than the query gain. Guided 3,000 is also
slightly below uniform 5,000 on query accuracy. On this exposed replay,
compositional guidance **does not establish a reliable accuracy or evaluation
budget advantage**, despite finding much higher-scoring training/support
programs.

Mean local search time was 2.39 seconds per person for guided and 2.35 for
uniform at 5,000 evaluations. These are workload-specific single-run times,
not a controlled hardware performance benchmark. The guided rule and its
hyperparameters were chosen during development, and the participant set has
been repeatedly inspected. The interval describes variation across these
people; it is not an independent validation interval.

The result redirects the next experiment. Increasing optimizer strength alone
can exploit noisy support scores. A useful search criterion must demonstrate
that its **ranking predicts disjoint future queries**, including unfavorable
participants, at matched measurement and compute cost. Candidate choices
should be assessed under independently resampled support repetitions and
session or contact perturbations before another guided-search variant is
called an advance. New people or another corpus are needed for an external
test of any rule selected using this replay.
