# Labeled support helps adaptation more than layout search

The [fixed-layout few-shot result](37-hyser-fewshot-heldout-result.md) showed
that one labeled new-session trial per gesture strongly improves Hyser
forearm recognition. We asked whether those same support labels could select
a better eight-site program from 1,000 feasible recorded layouts. This is a
post-hoc **development** replay on the previously explored 13 subjects, not
a new held-out cohort and not facial silent speech. Hyser groups repetitions
by gesture, so the support/query split is disjoint but not globally ordered.

The [program](../../software/python/scripts/hyser_calibrated_program_search.py)
uses only day-1 training and the first recorded day-2 repetition of each
gesture to select a layout. It compares day-1 three-fold CV, accuracy of the
day-1 classifier on those labeled day-2 support trials, and a half-weighted
average of the two. All selected layouts are then trained with identical
day-1 examples plus one support example per class at weight 1 and evaluated
on the remaining day-2 repetitions. The fixed program receives the same
adaptation and query set. Alpha 10, candidate seed 42, and whole-trial log
RMS/waveform-length features are shared across choices.

After observing subjects 01–05, we also tried a gate: choose the
support-selected layout only when its support accuracy exceeds the fixed
layout's support accuracy by at least 0.20; otherwise retain fixed. That
threshold is **post-hoc** relative to the first five people. Its next eight
subjects were run separately, but these subjects had already been used in
earlier model development, so this is not prospective validation.

| Selection rule | Mean adapted query accuracy, first 5 | Mean adapted query accuracy, next 8 |
| --- | ---: | ---: |
| Fixed program | 0.693 | 0.651 |
| Day-1 CV winner | 0.637 | 0.651 |
| Support-accuracy winner | 0.697 | 0.630 |
| Half CV, half support | 0.691 | 0.657 |
| Support improvement ≥0.20, else fixed | 0.720 | 0.651 |

The [first-five record](results/hyser-calibrated-program-search-dev05-v1.json)
and [next-eight record](results/hyser-calibrated-program-search-dev08-v1.json)
contain every chosen program, the fixed and selected support scores,
adapted/no-support query scores, support/query counts and source hashes. The
gate helped subjects 01 and 03 among the first five, but on the next eight
it helped two and harmed two; its mean paired gain over fixed rounded to
zero. The ungated support winner fell below fixed by 0.021 mean on the next
eight. A small support set can be valuable for model adaptation yet too
noisy for searching 1,000 programs.

This result is evidence against a naive large candidate search, not against
personalization in general. A better program search must account for
winner's curse and the effective number of independent layouts, ideally
using repeated, time-separated support blocks. For now the harness should
keep the fixed layout when design evidence is weak and direct expensive
evaluation toward calibration and robustness, where measured gains are much
larger. The facial-array corpus with ordered blocks remains the necessary
next data source for silent-speech claims.
