# A fast information proxy fails the next-day layout decision

The [synthetic differential-program pilot](39-differential-program-search-pilot.md)
proved that batched covariance scoring can make a hardware search fast. It did
not show that information scores select a layout that actually decodes better.
We tested the nearest available **recorded-channel analogue** on the public
[Hyser forearm HD-sEMG corpus](https://physionet.org/content/hd-semg/2.0.0/).
This is a post-hoc development audit on subjects 01–05; their day-2 results
were already known to earlier experiments. It is not a prospective validation
or a facial silent-speech result.

For each person, the new day-1 proxy computes class-balanced between-gesture
and within-gesture covariance of the two cached features per site (log RMS
and waveform length). It scores an eight-site program with a regularized
Gaussian log-determinant ratio. The same seeded pool of 1,000 feasible
programs receives the existing day-1 ridge cross-validation score. Both
rules select on day 1 only; the actual outcomes come from training a ridge
decoder on day 1 and scoring balanced accuracy on day 2. The fixed-grid
layout is an external baseline. The [result JSON](results/hyser-information-ranking-dev05-v1.json)
binds the audit to the source-record hashes and exact selected programs.

| Subject | Information rank correlation with day 2 | Information-selected day 2 | Day-1 CV-selected day 2 | Fixed day 2 |
| --- | ---: | ---: | ---: | ---: |
| 01 | +0.169 | 0.505 | 0.525 | 0.571 |
| 02 | +0.142 | 0.240 | 0.265 | 0.242 |
| 03 | −0.429 | 0.212 | 0.338 | 0.441 |
| 04 | −0.243 | 0.054 | 0.598 | 0.598 |
| 05 | −0.296 | 0.064 | 0.029 | 0.515 |

The information proxy selects a program below fixed for every person
(subject 02 is nearly tied). Its large losses on 03–05 rule it out as a
layout-selection policy. It is also unreliable as a cheap first-stage filter:
among the proxy's top 100 of 1,000 programs, the recall of the actual top
100 is 0.08, 0.13, 0.00, 0.10, and 0.00 for subjects 01–05. Random overlap
is about 0.10 in expectation; subjects 03 and 05 lose every actual top-decile
program. Even an ideal second-stage evaluator cannot recover candidates
removed by that filter. A high within-session class-separation score can
reward transient contact or gesture-block structure that fails across days.
That explanation is a hypothesis, not a diagnosed cause. The result also
does not falsify all covariance models: this proxy uses cached bipolar
features and no contact-level forward model, anatomical calibration, or
future-session labels.

The immediate harness rule is that **search speed cannot substitute for
decision validation**. Keep a fixed-layout fallback, measure candidate
ranking and shortlist recall on future sessions, and reject this proxy as
an aggressive filter under its current definition. The next
facial-array test should compare a shortlist against a fixed layout on
ordered, held-out blocks. A model with a good synthetic waveform or
information score but bad design ranking should abstain from hardware advice.

Reproduce from `software/python` with the verified local feature cache:

```powershell
.\.venv\Scripts\python.exe scripts\hyser_information_ranking.py --cache-dir F:\Datasets\OpenAlterEgo\Hyser\features-v1 --subjects 01 02 03 04 05 --candidates 1000 --output ..\..\docs\research\results\hyser-information-ranking-dev05-v1.json
```
