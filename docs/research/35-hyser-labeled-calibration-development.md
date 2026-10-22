# One- and two-shot new-session calibration: development result

**Status:** post-hoc development on the same 13 Hyser subjects used for
earlier model design. The [Hyser source](https://physionet.org/content/hd-semg/2.0.0/)
contains forearm gestures, so this tests a method for handling session drift,
not facial silent-speech performance.

The previous [temporal-feature pilot](34-hyser-temporal-features-pilot.md)
found no reliable gain from four within-trial time bins. We instead gave the
fixed eight-site ridge classifier a small amount of **labeled information
from the new session**. For each of the 34 nominal gestures present in day 2,
the first one or two recorded repetitions form a support set. All remaining
repetitions of those classes form a disjoint query set. The day-1 ridge alpha
was chosen by day-1 CV in the earlier pilot. The baseline and adapted model
are scored on the same query trials. Day-1 examples carry class-balanced
weight; each new-session support class receives 0.25, 1, or 4 times that
per-class weight. The [full development result](results/hyser-fewshot-dev13-v1.json)
records each subject, feature family, support size, weight, and query score.

| Feature family | Support per gesture | No support, same query | Weight 1 | Mean gain | People helped |
| --- | ---: | ---: | ---: | ---: | ---: |
| Whole trial | 1 | 0.425 | **0.693** | **+0.268** | 13/13 |
| Whole trial | 2 | 0.424 | **0.742** | **+0.318** | 13/13 |
| Four bins plus whole | 1 | 0.439 | **0.644** | **+0.205** | 13/13 |
| Four bins plus whole | 2 | 0.435 | **0.758** | **+0.323** | 13/13 |

With one support trial and whole-trial features, the smallest individual
improvement was +0.129. Weight 4 reached 0.696 mean with one shot and 0.770
with two, but these weights were explored after viewing development outcomes.
The frozen next-cohort primary rule will use **one support repetition per
gesture, whole-trial features, and weight 1**, which gives the smaller
calibration burden and a strong development signal.

This split has an important temporal limitation. Hyser presents all six
repetitions of one gesture together before moving to the next gesture.
Taking the first repetition of every class as support means that support
for later gestures was actually recorded after query trials of earlier
gestures. The test is a *within-session episodic support/query replay*,
not an online chronological calibration demonstration. Same-block contact
conditions and autocorrelation may make the task easier than a real wearable
that is reseated later. A future controlled acquisition should present the
whole calibration set first, then collect temporally separated query trials.

The method also consumes **34 labeled prompt responses** for one-shot
support, or 68 for two-shot support. Each Hyser trial is one second, so the
minimum signal time is 34 or 68 seconds, excluding prompts, pauses, fitting,
and user setup. That burden must travel with any accuracy claim. The result
does not imply zero-calibration generalization or open-vocabulary decoding.
