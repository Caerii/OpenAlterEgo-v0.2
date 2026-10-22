# Support-conditioned program search: method-check result

The [protocol and five-person development result](52-support-conditioned-program-search.md)
fixed the candidate grammar, support-conditioned selection rule, and
success criterion before this seven-person method check. The
[selection record](results/hyser-support-search-method-v1-selection.json)
was committed and pushed at `08d665f` before query accuracy was computed.
The [score artifact](results/hyser-support-search-method-v1-score.json)
binds to selection SHA-256
`a8bd5b3909065db5988f6315b2b8e52e65a354e46fb425c57842988f6cee0810`.

## Primary result

Each program used one labeled day-two trial per gesture with support
weight 4, then was scored on the remaining identical query trials.
The selected program was the best of 5,000 equal-cost complete programs
by the fixed average of day-one CV and day-one-to-support accuracy.

| Subject | Selected program | Fixed eight-site | Paired gain |
| --- | ---: | ---: | ---: |
| 06 | 0.841 | 0.829 | +0.012 |
| 07 | 0.806 | 0.741 | +0.065 |
| 08 | 0.687 | 0.671 | +0.016 |
| 17 | 0.708 | 0.714 | -0.006 |
| 18 | 0.750 | 0.706 | +0.044 |
| 19 | 0.800 | 0.830 | -0.030 |
| 20 | 0.782 | 0.635 | +0.147 |
| **Mean** | **0.768** | **0.732** | **+0.035** |

The predeclared success rule required mean paired gain at least +0.05,
five of seven people improving, and no loss worse than -0.05. It met
the latter two conditions but **failed the mean-gain threshold**.
The development mean gain was +0.059; the method-check gain was +0.035.
This is a promising signal for further investigation, not the claimed
large and reliable hardware advantage.

## Secondary scores

| Program | Mean query balanced accuracy |
| --- | ---: |
| Support-only selection | 0.773 |
| Half CV, half support selection (primary) | 0.768 |
| Day-one CV selection | 0.763 |
| Fixed eight-site shared-reference | 0.732 |
| Fixed mixed | 0.714 |
| Primary selection contacts, all shared-reference | 0.699 |

The primary selected program exceeded its own all-shared-reference
contact control by +0.068 mean, suggesting that measurement mode matters
in this *digital replay*. The support-only rule scored highest here, but
it was secondary and chosen from this result. It must not replace the
frozen primary rule without a new, genuinely independent evaluation.
The three selectors differ by about one percentage point in mean
accuracy, so this result does not isolate why the program choices help.

## Decision

Keep the fixed eight-site program as the strongest simple reference.
Preserve the support-conditioned selector as an exploratory method
candidate. The next useful experiments should quantify ranking stability
under support-trial resampling and channel perturbation, and test the
same decision rule on a dataset with independent participants, true
facial channels, or a chronologically separate session. Optimizing the
5,000-program search runtime is lower priority than establishing that
the chosen layout remains better after such changes.

This is an exposed-subject, forearm gesture benchmark. The first
classwise repetition is not a globally chronological calibration
sequence, and digital differences of preprocessed signals omit analog
contact and front-end behavior. The result does not validate facial
silent speech or an actual wearable.
