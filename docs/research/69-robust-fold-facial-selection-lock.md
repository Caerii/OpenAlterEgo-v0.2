# Fixed worst-fold facial selection check

The [exact search audit](68-exact-facial-search-regret-audit.md) found
many programs tied by mean forward-fold accuracy. We tested one fixed
alternative on FaceEMG-11 subjects 01–06: with equal six-channel cost
and beam width four, maximize the **minimum** of two forward-fold
balanced accuracies instead of their mean. Its mean later-block
accuracy was 0.897 versus 0.882 for the mean-fold rule. This development
result is retrospective.

Before running the comparison on subjects **07–12**, freeze:

- Features: existing 20-channel, eight-features-per-channel temporal
  caches, with recorded member hashes checked by the loader.
- Per-person selection folds: fit blocks 1–10, validate 11–15; fit
  1–15, validate 16–20. Balanced accuracy; ridge alpha 10.
- Search: six distinct recorded channels; beam width four; compare
  the mean of two fold scores to their minimum. Lexicographic ties
  follow the existing beam implementation. No choice from later blocks.
- Evaluation: refit on blocks 1–20, score blocks 21–30. Record every
  subject's choice and balanced accuracy, then the unweighted mean
  of subjects 07–12. The all-20-channel score is a costlier context
  reference. Report decreases as well as gains.

These participants' later-block outcomes were inspected in prior
research, so this local lock does not provide an independent subject
holdout. FaceEMG-11 is a facial **command** dataset recorded once per
person; this comparison cannot establish continuous silent speech,
session transfer, or a causal electrode-design effect.
