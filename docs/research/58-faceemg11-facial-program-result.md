# Facial six-channel search: future-block method check

The [protocol](57-faceemg11-facial-program-protocol.md) and
[source-bound selections](results/faceemg11-method06-v1-selection.json)
were pushed at `e430b38` before the later-block query scores were
computed. The [machine-readable score](results/faceemg11-method06-v1-score.json)
binds to selection SHA-256
`7cadb4b19c77789cf6c0da50f3dd29c7f6aaf09d7c75288a52acde720972f3f1`.

All programs use the same six-recorded-channel budget except the all-20-channel
reference. For each of six previously unscored people, the personal
program was selected from all 38,760 six-channel subsets using
three-fold CV within blocks 1–15. A shared six-channel program was
chosen using forward validation on six *different* development people.
Models were refit on blocks 1–20 and scored on blocks 21–30 of each
method-check person. The outcome is 11-class balanced accuracy.

| Person | Personal six | Shared six | Paired gain | All 20 |
| --- | ---: | ---: | ---: | ---: |
| 07 | 0.900 | 0.809 | +0.091 | 0.927 |
| 08 | 0.909 | 0.855 | +0.055 | 0.927 |
| 09 | 0.900 | 0.827 | +0.073 | 0.927 |
| 10 | 0.900 | 0.882 | +0.018 | 0.973 |
| 11 | 0.809 | 0.718 | +0.091 | 0.900 |
| 12 | 0.891 | 0.873 | +0.018 | 0.927 |
| **Mean** | **0.885** | **0.827** | **+0.058** | **0.930** |

The predeclared method-level rule required mean paired gain at least
+0.05, improvement for at least five of six people, and no loss below
-0.05. It passed: +0.0576 mean and six of six improved. A descriptive
paired 95% *t* interval across the six people is +0.0226 to +0.0926;
with this small sample its distributional assumption is untested.
The exact two-sided sign-test probability under exchangeable signs is
0.03125. These statistics do not turn a command dataset into a
silent-speech result.

Secondary mean scores: personal forward-validation six 0.885,
shared train-CV six 0.823, and a six-channel index-spread layout
0.815. Six selected channels retained 0.885 accuracy versus 0.930
with all 20, using 30% as many measured channels. The all-20 reference
is not an upper bound for every person, because additional features
can also hurt a regularized classifier.

## What this establishes

On real facial EMG, exhaustive individual channel selection can
outperform a shared equal-channel layout on later blocks for new
participants. This is a useful experimental advantage and a stronger
test of the program-search harness than the forearm simulations.

It does **not** establish silent speech recognition, reconstruction of
voice or emotion, a superior wearable geometry, stable performance
after device removal, or an analog-electronics gain. FaceEMG-11 contains
facial and ocular commands within each person's recording; its
channel names do not supply measured electrode coordinates. The source
[Zenodo record](https://zenodo.org/records/22137268) does not state a
reuse license, so raw and derived signal caches remain external to
the repository. The next transfer test should preserve this fixed
selection method and assess actual speech-related facial arrays across
sessions before design claims are broadened.
