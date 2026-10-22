# A smooth proper score does not rescue program selection

The [guided search comparison](78-guided-compositional-search-generalization.md)
found a large rise in internal selection score with little improvement on
disjoint later-day queries. We tested whether hard balanced accuracy was too
coarse an objective: many candidates tie, and a smooth score could distinguish
their errors and confidence.

The [batched scorer](../../software/python/openalterego/research/dense_programs.py)
now supports a class-balanced multiclass Brier score. It turns ridge outputs
fitted to one-hot labels into probabilities using a fixed, temperature-one
softmax, then computes `1 - Brier/2` per validation trial and averages by
class. A gesture absent from day-one training receives zero predicted
probability; that case occurs in this corpus and is covered by a focused test.
The original hard-accuracy mode remains the default.

The [comparison runner](../../software/python/scripts/hyser_brier_program_selection.py)
scores **identical seed-42 pools of 5,000 complete programs** using half
day-one three-fold CV and one quarter each of two separate labeled day-two
support repetitions. It selects once by hard accuracy and once by Brier.
Both chosen programs use the same day-one training, two support examples per
gesture, decoder, and disjoint day-two queries. The fixed eight-site layout
is a reference. The [result record](results/hyser-brier-program-selection-12-v1.json)
contains source hashes, support/query indices, chosen programs, and outcomes
for the same 12 previously exposed forearm participants.

| Selector | Mean later-day query balanced accuracy |
| --- | ---: |
| Hard balanced accuracy | **0.821** |
| Class-balanced Brier | 0.793 |
| Fixed eight-site layout | 0.773 |

Brier minus hard selection averages **−0.0281** across people, with a
10,000-draw participant-bootstrap interval **[−0.0630, +0.0055]**. Brier wins
for three, loses for eight, and ties for one. It still exceeds the fixed
layout by +0.0198 on average, but it is worse than the existing search rule.
This comparison does not justify replacing hard accuracy with Brier.

Temperature one is a fixed operational choice, not a calibrated probability
model. Because both the support examples and all 12 participants have been
inspected repeatedly, these results are retrospective diagnostics rather
than independent evidence about a future wearer. The Hyser signals are
forearm measurements with digitally derived differences; this experiment
does not establish facial or analog hardware performance.

The next search objective needs a stronger source of independent information,
not merely a smoother transform of the same small support set. In particular,
candidate rankings should be tested against different support repetitions,
sessions, and measured contact perturbations with a frozen rule before a
claimed gain is accepted.
