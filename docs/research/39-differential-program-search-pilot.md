# Differential-program search: fast proxy, unvalidated decision

The current search space is a **hardware program**: a subset of differential
contact pairs, subject to separate limits on physical contacts and analog
front-end (AFE) channels. This pilot uses 40 illustrative contacts, 38 possible
adjacent pairs on two chains, at most 14 active contacts and exactly eight AFE
channels. There are `C(38, 8) = 48,903,492` nominal eight-channel subsets
before the contact limit. The pair map is illustrative and has **not** been
verified against the [CSL-EMG_Array corpus](38-facial-array-hardware-program.md).

The [search implementation](../../software/python/openalterego/research/differential_search.py)
precomputes each scenario's full signal and noise covariance once. Candidate
programs then gather small principal submatrices and score them with batched
log determinants. Shared-contact noise correlations remain in those
submatrices. The objective is mean Gaussian information minus one standard
deviation across development scenarios. A beam search proposes programs;
random search with the same number of scored programs and a fixed layout are
comparators. This is an information proxy, **not** an EMG recognition metric.

## Reproducible synthetic pilot

Run from `software/python`:

```powershell
.\.venv\Scripts\python.exe scripts\differential_program_pilot.py --output ..\..\docs\research\results\differential-program-synthetic-v1.json
```

The committed [result JSON](results/differential-program-synthetic-v1.json)
records seeds, assumptions, platform, outputs and timing. Sixteen synthetic
scenarios drive selection; 64 scenarios generated with another seed assess
the selected programs. The lead fields are smooth region-local functions;
source strength and sensor noise vary lognormally, and two contacts per
scenario get additional noise. No parameters were fitted to facial EMG.

| Choice | Eight channels | Active contacts | Evaluation mean bits/sample | Risk objective bits/sample |
| --- | --- | ---: | ---: | ---: |
| Beam | 17, 31–37 | 10 | 4.144 | 3.850 |
| One matched-budget random search | 5, 6, 16, 17, 32–35 | 11 | 3.184 | 2.900 |
| Fixed illustrative layout | 1–3, 15–17, 31, 32 | 11 | 1.688 | 1.397 |

The beam scored 5,449 programs in 0.225 s on this machine. For 64 feasible
programs across the 16 development scenarios, batched covariance scoring
took 0.00512 s versus 0.508 s when each candidate rebuilt its covariance:
about 99× in this narrow, single-run benchmark. Numerical agreement of the
mean information score was within `8.9e-16` bits/sample. These figures
measure **only covariance scoring**, excluding scenario generation, model
fitting, real-data decoding, and other search overhead. The random comparison
uses one seed and equal scored-program count, not a distribution of outcomes
or exactly equal CPU time.

The selected program concentrates seven of eight channels on the short chin
chain. That may reflect the hand-built source and noise assumptions rather
than an anatomical insight. In particular, a Gaussian information objective
can reward signals that carry source variance even when those signals are
irrelevant to silent-speech labels. The synthetic held-out scenarios are
different draws from the **same generator**, so this result establishes
software correctness and search throughput, not design-ranking validity.

## Highest-value next performance work

1. **Decision validity:** obtain the official CSL array data and contact-pair
   metadata, hold later recording blocks out, and ask whether a layout's
   predicted improvement tracks actual decoding improvement after fixed
   calibration. Include fixed, simple empirical selection, and random
   baselines. A stronger covariance optimizer has no value if its rankings
   are wrong.
2. **Search efficiency:** use cheap covariance scores to shortlist programs,
   then spend decoding runs only on candidates whose ranking is uncertain.
   Reuse feature extraction and decoding folds across masks. Count whole
   experiment wall time and quality per compute-hour, not kernel speed alone.
3. **Model adequacy:** test multiple lead-field and noise families and an
   abstention rule that retains a fixed layout when predicted benefit is
   unstable. Introduce detailed field physics only when a measured residual
   identifies what the simpler model misses.

No current result supports a claim that these selected contacts improve a
physical wearable. An unrecorded contact location or reseating intervention
cannot be validated from an existing fixed array without new measurements.
