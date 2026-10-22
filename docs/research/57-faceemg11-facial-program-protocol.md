# Facial six-channel program search on ordered command blocks

The [FaceEMG-11 public archive](https://zenodo.org/records/22137268)
contains 12 people, 20 facial sEMG channels, and 30 ordered blocks per
person. Each block contains one trial from each of 11 facial or ocular
commands. It is a useful **facial measurement** benchmark, but it is
not silent speech, prosody, or an electrode-array coordinate map.
The Zenodo record currently does not specify a reuse license; raw data
and derived feature caches remain outside this repository. The archive
was downloaded with its published MD5 verified.

The [extractor](../../software/python/scripts/faceemg11_extract.py)
records source member SHA-256 and computes trial-local log RMS and
waveform-length features for each of 20 channels. It validates the
1 kHz, 1,500-sample trial format and the 30 complete command blocks.
No normalization uses later blocks. The classifier is the existing
class-balanced ridge with training-only scaling, alpha 10.

Search all **38,760** six-channel subsets. Blocks 1–15 are used to
fit and score a within-training three-fold CV selector. Blocks 16–20
provide a forward validation selector. After choices are frozen, refit
on blocks 1–20 and score blocks 21–30. This is within-person future-
block evaluation; there is no device removal or new recording session.

## Six-person development observation

| Program | Mean later-block balanced accuracy |
| --- | ---: |
| Personalized train-CV six-channel selection | 0.874 |
| Personalized forward-validation six-channel selection | 0.864 |
| Simple spread across channel indices, six channels | 0.791 |
| All 20 channels (higher contact cost) | 0.930 |

These are subjects 01–06 only. The [source artifact](results/faceemg11-program-search-dev06-v1.json)
contains their choices, source hashes, individual scores, and timings.
An exhaustive search took 8.3–12.6 seconds per person in this local
run. The index-spread reference is weak because channel names do not
specify physical location. A stronger comparator must be selected
from *other participants* before testing subjects 07–12.

## Frozen method check

The [two-stage runner](../../software/python/scripts/faceemg11_method_check.py)
uses subjects 01–06 to choose the best shared six-channel programs by
mean training CV and mean forward validation, without consulting their
blocks 21–30. It independently chooses a personal train-CV and personal
forward-validation six-channel program for subjects 07–12 using only
their blocks 1–20. It commits all choices and source/code hashes before
query scoring. The main comparison is personalized train-CV selection
versus the shared program selected by development-subject forward
validation. Report both shared selectors, personal forward selection,
index-spread six channels, and all 20 channels.

For a compelling six-channel personalization gain, require mean paired
gain of at least +0.05, improvement on at least five of six new people,
and no individual loss below -0.05 against the shared six-channel
program. Even if this passes, it establishes only better selection for
these facial command trials, not silent-speech wearable performance.

The fixed block order protects the final ten blocks from selection,
but the entire archive is available to this local process; this is a
code-and-record separation rather than an independent data custodian.
