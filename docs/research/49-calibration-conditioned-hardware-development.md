# Calibration changes the ordering of measurement programs

The [compositional search](48-compositional-search-development.md) failed
to predict next-day performance without new-session calibration. We kept
its choices fixed and gave every program the **same** first recorded
day-2 trial per gesture as labeled support, then scored only the remaining
trials. The [batched supported-program scorer](../../software/python/openalterego/research/supported_programs.py)
matches the existing single-program ridge implementation on identical
inputs. It applies the same day-1 scaling, class-balanced weighting and
alpha 10 to programs over 704 possible measured signals.

| Program | No support, same query | One shot, weight 1 | One shot, weight 4 |
| --- | ---: | ---: | ---: |
| Compositional beam choice | 0.234 | 0.666 | 0.735 |
| Equal-count random choice | 0.346 | 0.709 | 0.734 |
| Fixed mixed mode | 0.389 | **0.714** | **0.758** |
| Fixed eight-site shared-reference | **0.474** | 0.693 | 0.704 |
| Beam's contacts, all shared-reference | 0.233 | 0.543 | 0.573 |

These are five **development** people, averaged at the participant level.
The [result record](results/hyser-compositional-calibration-dev05-v1.json)
includes every person's score, 33–34 labeled support trials, 164–170
disjoint query trials and publisher-derived source hashes. In the weight-1
comparison, fixed mixed improves over fixed shared-reference for two
people and loses for three; its mean advantage is +0.021. At weight 4 it
improves all five people and has a +0.054 mean advantage, but this weight
was examined during development. The beam's much higher day-1 CV does not
translate to the best adapted decision.

The fixed mixed program consumes eight array contacts plus a common
reference and eight AFE channels, the same nominal counts as the fixed
eight-site comparison. Its digital differential channels were calculated
from publisher-preprocessed **forearm** signals, so this does not prove
analog circuit behavior or facial silent speech. Hyser records gesture
repetitions in class blocks: a first repetition from a later class occurs
after query trials from earlier classes. This is an episodic support/query
replay, **not** a globally chronological interaction test.

The development signal justifies one frozen comparison of the two fixed,
equal-contact programs on the seven other available Hyser people. It does
not justify selecting the best hardware layout from their outcomes or
claiming an unrestricted wearable advantage.
