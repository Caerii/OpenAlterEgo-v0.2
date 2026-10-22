# Dense-array search on real two-day EMG: Hyser pilot

**Status:** development method check on forearm gestures, not evidence for
facial silent speech or a new electrode position. The source is the
[Hyser high-density EMG corpus](https://physionet.org/content/hd-semg/2.0.0/),
which records 256 channels over four 8×8 forearm grids from 20 people on two
separate days. We downloaded the publisher's dynamic, preprocessed gesture
records for subjects 01–03, both sessions: 2,400 SHA-256-verified files,
1.28 GB. The publisher `SHA256SUMS.txt` digest is
`ab9f6b1b2f65f8f0041ff67f9d4ea28d417e402f95c68040dca71c4afe7e3bfe`.
The raw files and derived feature arrays stay outside Git. Six archived
`hyser-features-*.json` files in `results/` preserve the selected publisher
record-set hashes, channel names and sample counts. Hyser's source license is
Open Data Commons Attribution 1.0; cite its authors when using the data.

This dataset is useful because it makes high-dimensional *recorded-site*
search testable on real signals with a new-day outcome. It does not provide
face/neck anatomy or measurements at unrecorded locations. Its grid labels
specify topology, but no millimetre spacing is assumed here.

## Search protocol

Each trial becomes 512 features: log RMS and log waveform length for each of
256 WFDB-scaled channels. The loader checks channel order, first samples,
WFDB checksums, and the publisher's file hashes. The six selected sessions
contain 196–203 trials each, drawn from nominally 34 gesture classes; some
sessions lack a class. The classifier handles the actual class set in each
training session.

We search eight-site layouts. The unconstrained space has
`C(256, 8) = 409,663,695,276,000` subsets. A feasible layout must separate
sites on the same grid by at least two grid cells. Three stratified folds of
session 1 give the selection score. Batched ridge solves use shared weighted
Gram and target matrices, so each extension does not refit from raw trials.
After choices are written to exclusive local records, the same ridge rule is
fit on all of session 1 and scored on untouched *within this run* session 2.
These are development subjects; the algorithm was written without seeing
their next-day scores but may be revised before a separately held-out cohort.

Comparators are a fixed spatially spread design, one seeded random feasible
design, and the best of as many random full designs as greedy evaluates
candidate extensions (1,826–1,870 objectives). An all-256-channel fit is an
observed-information reference, not a wearable layout. Random full designs
cost more per evaluation than short greedy prefixes; both objective counts
and wall times are recorded. The [full result and selection records](results/hyser-program-search-dev-v1.json)
give exact sites and source hashes.

| Subject | Greedy CV | Greedy next day | Budgeted random next day | Fixed-grid next day | All 256 next day |
| --- | ---: | ---: | ---: | ---: | ---: |
| 01 | 0.838 | 0.273 | 0.439 | **0.571** | 0.143 |
| 02 | 0.951 | **0.274** | 0.234 | 0.242 | 0.034 |
| 03 | 0.914 | 0.035 | 0.268 | **0.441** | 0.045 |
| **Mean** | **0.901** | **0.194** | **0.314** | **0.418** | **0.074** |

Chance balanced accuracy is near 1/34, subject to missing classes in a day.
The next-day collapse is especially severe for subject 03. The fixed grid wins
for two of three people, and budgeted random wins over greedy for those same
two. Fitting 256 channels with only about 200 training trials severely
overfits under this simple ridge setting; it is not a valid upper bound on
information available from the array. Greedy selection took 0.46–0.90 seconds
per subject after feature extraction, while budgeted random took 0.86–1.28
seconds on this single Windows host. Timings are exploratory, not controlled
hardware benchmarks.

## What the failure establishes

The current objective can find extremely discriminative **same-day** channels
quickly. It does not find channels that transfer to the next day. Increasing
the number of program evaluations or making the solver faster would optimize
the wrong target more efficiently. The fixed grid suggests that spatial
coverage is a useful *hypothesis* to test; it does not prove the cause of
transfer. Grid placement, contact, muscle recruitment, and source/task shift
remain confounded.

The next candidate rule will enforce balanced coverage of the four grids and
compare against a matching random search. Subjects 01–03 remain development
data. We will freeze the revised rule, then acquire subjects 04–05 and use
their second days as a separate cohort gate. That gate tests a search method
on forearm EMG. Translating it to a wearable silent-speech design still
requires spatial facial recordings and, eventually, measured electrode
interventions.

The coverage constraint was tested on the same development trio before that
gate. It caps each grid at two sites, yielding exactly two per grid in an
eight-site design. Next-day balanced accuracy for coverage greedy was
0.370, 0.281, and 0.030 on subjects 01–03, against fixed-grid 0.571,
0.242, and 0.441. The [development result](results/hyser-program-search-coverage-dev-v2.json)
records all choices and source hashes. Coverage helped subjects 01 and 02
relative to unconstrained greedy but did not solve subject 03. This is the
complete development evidence used to select the
[frozen subjects 04–05 protocol](30-hyser-heldout-coverage-protocol.md).

## Reproduction

Run from the repository root with `PYTHONPATH=software/python` and the Python
research dependencies installed. First download the publisher's `RECORDS` and
`SHA256SUMS.txt` from
`https://physionet-open.s3.amazonaws.com/hd-semg/2.0.0/` into an external
`F:\Datasets\OpenAlterEgo\Hyser` directory. Then run:

```text
python -m scripts.hyser_acquire --root F:\Datasets\OpenAlterEgo\Hyser --subjects 01 02 03 --sessions 1 2 --workers 16
python -m scripts.hyser_feature_cache --root F:\Datasets\OpenAlterEgo\Hyser --cache-dir F:\Datasets\OpenAlterEgo\Hyser\features-v1 --subjects 01 02 03
python -m scripts.hyser_program_search --cache-dir F:\Datasets\OpenAlterEgo\Hyser\features-v1 --subjects 01 02 03 --output docs\research\results\hyser-program-search-dev-v1.json
```

The final command uses exclusive selection records and refuses to overwrite
the archived result. Use a new output directory when reproducing it.
