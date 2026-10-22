# Compute scaling for sparse search in a dense channel space

The [compositional search](61-facial-search-compute-frontier.md)
was measured on 20 recorded facial channels. To check whether the
same scoring machinery is computationally plausible for the
40-contact, approximately 38-recorded-channel [CSL-EMG_Array corpus](https://www.uni-bremen.de/en/csl/research/silent-speech-communication/csl-emg-array-corpus),
we timed the unchanged beam-4 search and batched ridge scorer with
20, 40, and 80 candidate channels. Each synthetic matrix had 165
examples, 11 balanced classes, eight features per channel, and three
stratified scoring folds. The solver was warmed once per size and
then timed three times.

| Candidate channels | Six-channel combinations | Programs evaluated | Median local search time |
| ---: | ---: | ---: | ---: |
| 20 | 38,760 | 343 | 0.124 s |
| 40 | 3,838,380 | 761 | 0.366 s |
| 80 | 300,500,200 | 1,604 | 0.632 s |

The [benchmark runner](../../software/python/scripts/dense_search_scaling.py)
and [full timing artifact](results/dense-search-scaling-local-v1.json)
record each repetition, stage counts, and machine details. These
are local timings, not portable throughput promises. The Gaussian
matrices have **no biological meaning**, and the search scores on
them have no accuracy or hardware-design interpretation.

This removes one compute concern: enumerating millions of full
six-channel subsets is unnecessary for a first CSL array search.
It does **not** show that beam-4 finds the best layout in a
40-candidate-channel space, or that a channel subset is a feasible electrode
circuit. The next meaningful test must supply real synchronized
speech EMG, time-ordered evaluation, and explicit contact,
reference, AFE, and placement constraints. A 40-contact array
measurement can support subset search over its *recorded* channels;
it cannot validate arbitrary unmeasured electrode locations.
