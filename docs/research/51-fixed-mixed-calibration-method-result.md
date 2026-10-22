# Fixed mixed measurement with calibration: method-check result

The [protocol](50-fixed-mixed-calibration-method-protocol.md) and
[selection record](results/hyser-fixed-mixed-calibration-v1-selection.json)
were committed and pushed at `6bfb361` before this comparison was scored.
The [machine-readable score](results/hyser-fixed-mixed-calibration-v1-score.json)
binds to selection SHA-256
`76cd606f0a0d904e76620d292733d60f71bd5e541862991591d5ec78b0bbe8d3`.

## Primary result

One labeled day-2 repetition per gesture was used for calibration, with
support weight 4. Remaining day-2 repetitions were the query set. Scores
are class-balanced accuracy on identical support and query trials.

| Subject | Fixed mixed | Established fixed eight-site | Paired gain |
| --- | ---: | ---: | ---: |
| 06 | 0.771 | 0.829 | -0.059 |
| 07 | 0.729 | 0.741 | -0.012 |
| 08 | 0.606 | 0.671 | -0.065 |
| 17 | 0.702 | 0.714 | -0.012 |
| 18 | 0.747 | 0.706 | +0.041 |
| 19 | 0.770 | 0.830 | -0.061 |
| 20 | 0.676 | 0.635 | +0.041 |
| **Mean** | **0.714** | **0.732** | **-0.018** |

The predeclared rule required mean gain at least +0.05, gains for at
least five of seven people, and no loss worse than -0.05. It failed all
three conditions: mean gain was -0.018, two people improved, and three
lost more than 0.05. The +0.054 mean gain on the five development people
did not generalize to this method-check set.

## Secondary comparisons

| Support weight | Fixed mixed | Established fixed eight-site | Same contacts, all shared reference |
| --- | ---: | ---: | ---: |
| 0 | 0.406 | 0.464 | 0.361 |
| 1 | 0.683 | 0.697 | 0.618 |
| 4 | 0.714 | 0.732 | 0.669 |

The mixed measurement outperformed the all-shared-reference signals
from its *own contacts* by +0.045 at weight 4, but the established
eight-site layout remained better by +0.018. This supports a narrow
measurement-mode observation, not a better complete design. Calibration
itself produced a large absolute gain for all three programs, but the
first support trial for each gesture is not a globally chronological
online calibration session.

## Decision

Do not promote the fixed mixed program or its development effect as a
hardware win. Keep the established fixed eight-site layout as the
reference for this Hyser task. The next program-search objective should
predict accuracy *after* calibration on a different session, with a
strong fixed-layout baseline, uncertainty, and a full-design physical
cost. A large day-1 search score or realistic synthetic waveform is not
enough. Reuse no claim of independent participant validation here:
these subjects and day-2 recordings were exposed in earlier Hyser work.

Hyser is a forearm gesture dataset, and these differences were made
digitally from publisher-preprocessed channels. They do not validate
facial silent speech, analog differential electronics, electrode fit,
prosody, or a wearable.
