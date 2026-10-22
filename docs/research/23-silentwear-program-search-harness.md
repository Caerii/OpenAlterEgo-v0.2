# Budgeted search over recorded SilentWear channels

The question for this pilot is whether a cheap approximation can choose better
six-channel virtual montages per unit of computation than an equally budgeted
random search. A virtual montage keeps a subset of electrodes that were already
recorded. It does **not** predict the effect of moving an electrode, changing
contact impedance, or constructing a new wearable.

## Search and evidence boundary

[`program_search.py`](../../software/python/openalterego/research/program_search.py)
defines 3,003 six-channel subsets of the 14 recorded channels, two feature
normalizations, and three logistic-regression regularization settings: 18,018
programs. The first stage scores all 6,006 subset/normalization structures
using a batched ridge proxy built from shared sufficient statistics. It takes
the top 16 structures into full logistic fits. Four finalists per strategy
also try the other two regularization values. A comparator samples 16 programs
from the canonical program list independently of the proxy ranking and gets
the same 24 full fits for selection. The two strategies may happen to overlap.

SilentWear S01–S03 use session 1 for fitting, session 2 for selecting, and
session 3 only after both choices are recorded. S04 remains sealed. The CLI
checks the source manifest hash on each feature-cache miss. The cached event
arrays live outside Git. It writes an exclusive local selection record before
loading session 3, then an atomic cohort result. This is a useful ordering
guard, **not** an independent immutable preregistration: these public sessions
have been explored before, and the records are local files.
Feature normalization uses the complete unlabeled recording batch, so this is
an offline evaluation and does not establish streaming performance.

Reproduce from the repository root with the dataset environment installed:

```powershell
$env:PYTHONPATH = 'F:\Github\hardware-open-alter-ego-v0-2-pilot\software\python'
F:\Datasets\OpenAlterEgo\.venv\Scripts\python.exe software/python/scripts/silentwear_program_search.py `
  --root F:\Datasets\OpenAlterEgo\SilentWear `
  --subjects S01 S02 S03 `
  --cache-dir F:\Datasets\OpenAlterEgo\SilentWear\event-cache-v1 `
  --proxy ridge --seed 42 `
  --output docs/research/results/silentwear-program-search-ridge-v4.json
```

The [cohort result](results/silentwear-program-search-ridge-v4.json) and its
three adjacent selection records preserve all stage scores, selected programs,
source hashes, and wall/CPU times. `--resume` skips completed subjects when
the configuration matches; it cannot change a recorded choice.
The saved v4 artifact was produced at commit `d40a02f`; later code adds a
bidirectional mode and writes v5 artifacts. Use that commit to reproduce the
v4 protocol; timing and registration timestamps will differ.

## Exploratory result

Balanced accuracy is the mean class recall across the nine event labels. The
numbers below are **one seed and three development subjects**, not a population
estimate. Both strategies have the same full-fit budget.

| Subject | Ridge-ranked selection | Ridge-ranked next session | Random selection | Random next session | Search wall time |
| --- | ---: | ---: | ---: | ---: | ---: |
| S01 | 0.3697 | 0.2416 | 0.3689 | 0.2213 | 29.1 s |
| S02 | 0.3795 | 0.2780 | 0.3331 | 0.2456 | 21.0 s |
| S03 | 0.3639 | 0.2726 | 0.3412 | 0.2738 | 27.4 s |
| Mean | 0.3710 | 0.2641 | 0.3477 | 0.2469 | 25.8 s |

The ridge screening advantage on the next session is 0.0172 balanced-accuracy
points on this tiny cohort; it loses to random on S03. The mean selected
ridge program drops 0.107 points from session 2 to session 3. A cheaper
nearest-centroid screen in the separate [exploratory result](results/silentwear-program-search-centroid-v3.json)
averaged 0.2554 on session 3, versus 0.2641 for ridge and the identical
0.2469 random comparator. The difference between screening methods is smaller
than the session shift. Timings are single Windows-host runs with warm feature
caches and are not controlled speedup estimates.

This is a functioning search instrument, not an overwhelming advantage. The
observed improvement is too small and variable to justify claims about a
physical electrode design or a 10–1000× gain in useful discovery. Reusing
these session-3 results to choose the next variant makes them development data.

## What the physics says to measure next

The observed feature vector is a mixture of motor-unit sources filtered by
volume conduction, electrode geometry, contact, analog acquisition, and
movement. A channel subset is an observation intervention on an already
recorded montage; it cannot separate these latent causes. Several physical
explanations may fit the same recordings. The practical target is prediction
under a *measurable change*, especially removal and replacement across
sessions, rather than a plausible reconstruction of hidden anatomy.

1. Quantify class information and covariance stability for each channel and
   channel pair across sessions. Estimate the session-to-session noise floor
   and redundant directions before enlarging the search grammar.
2. Evaluate selection rules over multiple development subjects, seeds, and
   session orders with an untouched final evaluation unit. Track confidence
   intervals, ranking agreement, and **decision regret** against attainable
   recorded subsets, not only average classifier accuracy.
3. Search for invariants: features and normalizations whose class-conditional
   relationships survive a new session. Penalize session-sensitive programs
   during selection. A more expensive proxy is useful only if it predicts the
   future-session ordering more reliably per unit cost.
4. Add a constrained generative layer for electrode displacement, contact
   impedance, and anatomy only when public data or later measurements can
   distinguish its predictions. A high-fidelity field solver is premature if
   no intervention data can validate its layout ranking.

The immediate scientific focus is therefore transport across sessions and
calibrated design decisions. Faster exhaustive scoring is helpful, but it
does not create information absent from the measurements.
