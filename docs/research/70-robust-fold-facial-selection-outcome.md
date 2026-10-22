# Worst-forward-fold facial selection outcome

The [rule and subject set](69-robust-fold-facial-selection-lock.md) were
committed and pushed at `7a86abb` before this runner scored subjects 07–12.
The [record](results/faceemg11-robust-fold-posthoc07-12-v1.json) retains
source hashes, channel choices, both forward-fold scores, search budgets,
and later-block results. These later blocks were exposed in previous
research; this is a **retrospective consistency check**, not a new holdout.

| Subject | Mean-fold choice | Worst-fold choice | All 20 channels |
| --- | ---: | ---: | ---: |
| 07 | 0.873 | 0.873 | 0.918 |
| 08 | **0.891** | 0.882 | 0.955 |
| 09 | 0.900 | 0.900 | 0.991 |
| 10 | 0.964 | 0.964 | 0.982 |
| 11 | 0.900 | **0.927** | 0.982 |
| 12 | 0.909 | **0.927** | 0.945 |
| Unweighted mean | 0.906 | **0.912** | 0.962 |

The worst-fold objective raised mean later-block balanced accuracy by
**0.0061** over the original mean-fold objective. It helped two subjects,
hurt one, and had no measured effect for three. It did not close the
substantial gap to the costlier all-20 reference. On development subjects
01–06, the corresponding gain had been 0.0152; the smaller second-group
gain is a warning against treating that development improvement as a
stable estimate.

The result supports retaining per-fold scores and considering temporal
robustness in the program search harness. It does **not** establish an
overwhelming advantage, a better physical montage, or even a reliable
gain across new people. A more discriminating decision objective needs
recording-session variation, calibration and contact information, and
evaluation on genuinely unexamined speakers or sessions. The existing
facial command corpus cannot test month-scale wearable reseating or
silent-speech decoding.
