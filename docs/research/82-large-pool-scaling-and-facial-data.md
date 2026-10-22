# A 100,000-program scaling audit and the facial-data boundary

The [rank-transport audit](80-mixed-program-rank-transport.md) showed that
the main bottleneck is predicting which measured-signal program works on
later trials. We nevertheless tested whether much broader sampling changes
the engineering decision or the compute picture. The
[large-pool runner](../../software/python/scripts/hyser_candidate_scale_audit.py)
uses a [prefix-preserving sampler](../../software/python/openalterego/research/uniform_program_pool.py)
to generate 100,000 unique **complete** programs from the same physical
grammar: one adjacent two-contact edge per each of four forearm grids, two
signals per edge, eight array contacts plus a common reference and eight
front-end channels. The estimated grammar has about 12.7 billion programs.

The nested 5,000, 20,000, and 100,000 choices use exactly the same seed-42
candidate stream. All are scored by half day-one three-fold accuracy plus
one quarter each from two separate labeled day-two support repetitions.
Only after all choices are fixed does the runner fit the same supported
decoder and score disjoint day-two query trials. The
[result record](results/hyser-candidate-scale-100k-3-v1.json) retains the
source hashes, support/query indices, programs, selection scores, outcomes,
and component timings. The three people were chosen from an **already
exposed** forearm cohort; this is a scaling diagnostic, not validation.

| Subject | 5,000 query accuracy | 20,000 | 100,000 | Fixed eight-site |
| --- | ---: | ---: | ---: | ---: |
| 01 | 0.841 | 0.848 | **0.856** | 0.848 |
| 05 | 0.772 | 0.721 | **0.831** | 0.721 |
| 18 | **0.860** | 0.772 | 0.850 | 0.779 |
| Mean of these three | 0.824 | 0.780 | **0.846** | 0.783 |

Every selection score rises or stays level with pool size because each
larger pool contains the earlier one. Later-day accuracy is **not monotone**.
The 100,000-program choice is +0.021 mean above the 5,000-program choice
on these three people, with one loss; no generalization claim follows from
three exposed subjects. The 20,000-program checkpoint loses −0.044 mean
relative to 5,000. Optimizing the measured selection score harder can pick
a worse later-day device program.

On this Windows machine, sampling 100,000 programs took 2.0–2.3 seconds
per person. Scoring all programs took 47.7–49.7 seconds for three day-one
CV folds and 10.8–13.3 seconds for each of the two support transfers:
approximately **71–74 seconds of scoring per person**. These are local
single-run observations, not a controlled portable speed benchmark. The
batched evaluator can explore a large fraction of a million-program pool
without a deep model fit for every candidate, but faster optimization of
the present objective alone is unlikely to deliver the sought advantage.

## Directly usable facial data

We checked primary public sources on 2026-10-22. The
[CSL-EMG_Array publisher](https://www.uni-bremen.de/en/csl/research/silent-speech-communication/csl-emg-array-corpus)
describes 40 facial array channels, 12 sessions from eight speakers,
including four silent sessions, with an 8.8 GB download under CC BY 4.0.
Its download endpoint requests a name, institution and email, then sends
an individual link. No corpus download was attempted through that form.
The new [emg2face paper](https://arxiv.org/abs/2610.09304) reports 64
facial channels from 25 people and expressive facial-motion targets, but
we found no linked public recording download in the paper. It concerns
facial expression rather than silent-speech content. The directly
downloadable [FaceEMG-11 release](https://zenodo.org/records/22137268)
offers 20 facial channels from 12 people, but one recording per person and
command labels, so it cannot test day-to-day reseating. The existing
[SilentWear release](https://huggingface.co/datasets/PulpBio/SilentWear)
offers multi-day silent commands at 14 channels and is already represented
in this repository's research.

The next scientific gain requires a selector whose ranking transfers to
new facial sessions at a fixed calibration and channel budget. The new
sampler makes wider search feasible, while the evaluated data show why
search scale, physical realism, and predictive validity must be checked
separately. The forearm digital differences still do not measure facial
contact impedances, true analog differential acquisition, or a wearable's
performance after reseating.
