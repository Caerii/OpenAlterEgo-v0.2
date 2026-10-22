# Does forward-block selection improve sparse facial programs?

The [fast compositional search](61-facial-search-compute-frontier.md)
selected channels with shuffled trial folds. We tested whether a
time-ordered selection objective chooses programs that work better on
later blocks. This is a **retrospective development diagnostic** on
FaceEMG-11 facial and ocular commands. Query blocks 21–30 have already
been viewed in earlier work; this is not a fresh holdout.

The [new runner](../../software/python/scripts/faceemg11_forward_search.py)
uses the same 20-channel, four-bin feature cache, six-channel beam-4
program grammar, balanced ridge scorer (alpha 10), and final fit on
blocks 1–20 for both choices. Shuffled CV selects on blocks 1–15
with three stratified folds. The forward objective averages two
strictly time-ordered folds: fit blocks 1–10 and validate 11–15;
fit 1–15 and validate 16–20. No query block participates in selection.

| Selection | Mean later-block balanced accuracy, subjects 01–06 |
| --- | ---: |
| Shuffled CV, six channels | 0.9045 |
| Forward blocks, six channels | 0.8818 |
| All 20 channels, reference | 0.9485 |

Forward selection improved one of six people and worsened five.
The complete [per-person artifact](results/faceemg11-forward-search-dev06-posthoc-v1.json)
records programs, source-member hashes, selection scores, search
times, and later-block scores. It was generated with:

```powershell
cd software/python
.venv/Scripts/python.exe -m scripts.faceemg11_forward_search `
  --coarse-cache F:\Datasets\OpenAlterEgo\FaceEMG-11\features-v1 `
  --temporal-cache F:\Datasets\OpenAlterEgo\FaceEMG-11\features-four-bin-v1 `
  --subjects 01 02 03 04 05 06 --beam-size 4 `
  --output ../../docs/research/results/faceemg11-forward-search-dev06-posthoc-v1.json
```

The forward objective uses less fitting data in its early fold and
is more variable. This experiment does not establish why it lost.
The result argues against adopting it as a default based on
plausibility alone. FaceEMG-11 has only one recording per person,
no physical reseating measurement, and no silent speech. The next
decisive test needs multiple *recording sessions* with speech labels
and enough recorded channels to compare wearable-sized programs.
The [CSL-EMG_Array corpus](https://www.uni-bremen.de/en/csl/research/silent-speech-communication/csl-emg-array-corpus)
provides that combination, subject to its emailed download link.
