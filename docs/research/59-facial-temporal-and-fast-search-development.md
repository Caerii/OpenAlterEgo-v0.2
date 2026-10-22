# Temporal representation and fast facial program search

The [facial six-channel result](58-faceemg11-facial-program-result.md)
used only whole-trial log RMS and waveform length. To test whether its
apparent hardware-selection gain depended on that representation, we
computed four consecutive 375-sample bins of both features for each
channel from the same [FaceEMG-11](https://zenodo.org/records/22137268)
raw trials. The [extractor](../../software/python/scripts/faceemg11_temporal_extract.py)
keeps source member hashes and trial/block alignment. Feature caches
remain outside the repository.

## Frozen-program temporal replay

We retained the previously frozen six-channel programs and model split:
fit on blocks 1–20, score blocks 21–30. This is **post hoc** because
those query blocks had already been exposed. The richer model used the
same ridge alpha 10 and training-only scaling.

| Program | Two whole-trial features/channel | Eight temporal features/channel |
| --- | ---: | ---: |
| Personalized six-channel | 0.885 | 0.924 |
| Shared six-channel | 0.827 | 0.905 |
| All 20 channels | 0.930 | 0.962 |

The personalized-minus-shared difference shrank from +0.058 to +0.020.
Temporal information improved the shared hardware program more than
the personalized one. Thus the original channel-selection gain partly
compensated for a weak representation. The
[per-person replay](results/faceemg11-four-bin-method06-posthoc-v1.json)
includes individual decreases as well as increases.

## Two-stage and compositional search

We compared three ways to choose six channels using training-block CV
on subjects 01–06. The exact temporal search scores all 38,760 six-
channel programs. The [two-stage search](../../software/python/scripts/faceemg11_multifidelity_search.py)
scores all programs using two whole-trial features per channel, then
reranks the top 128 with temporal features. The
[stepwise search](../../software/python/scripts/faceemg11_stepwise_search.py)
adds one channel at a time and retains a beam of 16 programs per stage,
scoring partial programs with the temporal model.

| Search | Mean temporal train CV | Mean later-block query | Mean search time | Program evaluations |
| --- | ---: | ---: | ---: | ---: |
| Exact temporal | 0.921 | 0.902 | 21.08 s | 38,760 |
| Two-stage, 128 finalists | 0.911 | 0.921 | 7.17 s | 38,760 cheap + 128 rich |
| Stepwise beam 16 | 0.917 | 0.912 | 0.39 s | 1,123 rich, mostly partial |

These are exploratory wall times on one local machine, with no
controlled contention or warmed execution protocol. Stepwise beam 16
was about **54× faster** than exact temporal scoring in this run, and
its mean CV score was 0.004 lower. Its mean query score was 0.011
higher on the six development people, but that is not a stable
generalization estimate. The [exact-search comparisons](results/faceemg11-exact-temporal-dev03-v1.json)
and [remaining three](results/faceemg11-exact-temporal-dev03b-v1.json)
record per-person timings and choices. The
[two-stage artifacts](results/faceemg11-multifidelity-dev06-v1.json)
and [beam artifact](results/faceemg11-stepwise-dev06-v1.json)
record their own source hashes and scores.
On the six already-exposed method subjects, temporal reranking of 128
finalists added only +0.006 mean query accuracy over using the coarse
winner with the same temporal decoder; the
[post hoc artifact](results/faceemg11-multifidelity-method06-posthoc-v1.json)
records all six outcomes.

Next, freeze beam width 16 and replay it on subjects 07–12 without
altering the selection rule. Their later-block outcomes were already
viewed for other programs, so this will remain a **method-specific
post hoc check**, not a new independent holdout.

These are facial/ocular commands within a single recording. They do
not validate speech phonemes, expressive voice reconstruction,
placement after reseating, analog contact effects, or a wearable.
