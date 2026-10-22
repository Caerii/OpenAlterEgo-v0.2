# Fixed-layout representation controls: Torchhd and a small temporal CNN

This is a retrospective development comparison on public SilentWear recordings.
It tests signal representations on **already recorded** channels. It does not
predict signals at new electrode sites, test a physical intervention, or
validate a wearable. Session 3 has been inspected in earlier project work and
is not a fresh holdout.

## Protocol

- Subjects: S01–S03; S04 is excluded from these development runs.
- Fit on sessions 1 and 2 of each person; score session 3 of that person.
- Fixed six recorded channel indices: `(0, 2, 5, 8, 11, 13)`; these indices
  do not encode anatomical spacing.
- Outcome: nine-class event balanced accuracy. These are isolated, centered
  400-sample event windows, so this does not measure streaming detection or
  end-to-end speech recognition.
- The feature logistic reference uses recording-robust normalized log RMS and
  waveform-length features. The Torchhd control uses the same features and an
  additional train-only standard scaler. The CNN uses raw windows with a
  different, unlabeled per-recording amplitude normalizer. Their differences
  therefore include preprocessing as well as classifier architecture.

## Torchhd library trial

We used [Torchhd](https://github.com/hyperdimensional-computing/torchhd)
5.8.4, specifically its MAP `Projection`, `normalize`, and cosine `Centroid`
components. The projection is random and the centroid is formed in one pass;
there is no gradient training. Widths 512, 2048 and 8192 were specified
before this run. Seed 42, PyTorch 2.11.0+cpu, eight CPU threads. Fit times
exclude file loading and Python import, but include train feature scaling,
projection and class-prototype addition. Inference times include test scaling,
projection and cosine classification over 1,605 events. Timings are exploratory
wall-clock measurements on a concurrently used machine, not controlled speed
benchmarks.

| Person | Logistic fixed spread | Torchhd 512 | Torchhd 2048 | Torchhd 8192 |
| --- | ---: | ---: | ---: | ---: |
| S01 | 0.3558 | 0.2481 | 0.2479 | 0.2466 |
| S02 | 0.2588 | 0.2199 | 0.2191 | 0.2143 |
| S03 | 0.2783 | 0.2590 | 0.2531 | 0.2553 |
| **Mean** | **0.2976** | **0.2423** | **0.2400** | **0.2387** |

The 512-dimensional encoder required 0.11–0.40 seconds to fit and uses about
43 KiB of float32 projection and class-prototype weights. Increasing width
did not improve future-session accuracy in this fixed setup. The existing
feature logistic control remains better on every person; high dimension by
itself did not create useful invariance. This is a result about *this encoder*,
not a rejection of HDC in general. Structured binding, temporal n-grams,
channel dropout, and online prototype adaptation are still testable, but each
needs a frozen selection split and an independent evaluation before any claimed
gain. Under this exposed evaluation, the next useful experiment is to improve
session invariance and inspect class confusions, rather than sweep more widths
against session 3.

The executable is `software/python/scripts/silentwear_torchhd.py`. It reads
the SHA-associated external feature cache and records subject manifest hashes,
source files, accuracy, fit/inference time, model size and tool versions. The
full run JSON is archived in
`docs/research/results/silentwear-torchhd-fixed-v1.json`.

## Temporal control

The raw-window cache and compact three-stage CNN provide an independent
nonlinear control using the same sessions and indices. Its normalization uses
unlabeled samples from each complete recording, so it is an offline result.
The CNN is not a causal or streaming implementation. It used a fixed six-epoch
budget, seed 42, class-weighted cross entropy, AdamW and PyTorch CPU; each
person had 3,210 training and 1,605 test events.

| Person | CNN balanced accuracy | Fit seconds, excluding data load |
| --- | ---: | ---: |
| S01 | 0.2210 | 91.0 |
| S02 | 0.1986 | 60.7 |
| S03 | 0.2762 | 53.2 |
| **Mean** | **0.2319** | |

The CNN learned a decreasing training loss but did not beat the fixed logistic
reference for any person. The six-epoch budget and different amplitude
normalization limit what this says about temporal representation learning;
larger networks are not warranted from this result alone. The full run JSON is
archived in `docs/research/results/silentwear-raw-cnn-v1.json`. The executable and
reproducible raw-window extraction are
`software/python/scripts/silentwear_raw_cnn.py` and
`software/python/scripts/silentwear_window_cache.py`. First-file extraction
parity against the existing feature cache matched labels exactly, with
maximum absolute feature difference `2.39e-7`.

## Decision

Keep the feature logistic model as the reference for fixed-layout search.
Neither this Torchhd projection-centroid control nor this fixed-budget CNN
demonstrated a several-fold representation advantage. Next, test a specific
session-shift hypothesis using train/validation data only: compare confusion
matrices and feature drift across sessions, then design a channel-aware
temporal encoder or structured HDC binding to address the observed failure.
Freeze that choice before another independent participant/session evaluation.
None of these classifiers substitutes for an intervention-validated forward
model of unrecorded electrode positions.
