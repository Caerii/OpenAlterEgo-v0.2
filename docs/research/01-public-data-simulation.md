# Public-data and simulation-first research lane

Status: active offline work. No new participant recordings or physical device
experiments are required for this lane.

## What can be learned now

Use freely available recorded EMG to validate preprocessing, decoder transfer,
session drift, channel subset selection, and uncertainty. Apply the simulator
to controlled perturbations and ask whether those perturbations predict
existing held-out recordings. This can reject bad models and improve software
and experimental decisions. A subset of already recorded channels cannot
predict what an electrode would measure at a new location; synthetic feature
drift has no measured contact-impedance calibration here.

## Public data queue

| Corpus | Access and rights checked 2026-10-21 | Use |
| --- | --- | --- |
| [SilentWear](https://huggingface.co/datasets/PulpBio/SilentWear) | Direct Hugging Face download; dataset card says Apache-2.0; 4 participants, 3 days, silent/vocalized commands, 14 recorded channels. Full repository is about 10.9 GB. | First virtual-montage, day-transfer and synthetic-drift benchmark. |
| [Gowda small vocabulary](../gowda/datasets/01-small-vocab.md) | Existing project benchmark; OSF data provenance and rights recorded in the Gowda legal notes. | Pinned sim-to-real transfer and phoneme/decoder diagnostics. Do not select models on its test partition. |
| [CSL-EMG_Array](https://www.uni-bremen.de/en/csl/research/silent-speech-communication/csl-emg-array-corpus) | University describes CC BY 4.0, 8 speakers, 12 sessions, approximately 8.8 GB; its download link is emailed after an address is entered. | Later spatial array and cross-speaker test after independent import. |
| [Gaddy Silent Speech EMG](https://zenodo.org/records/4064409) | Open Zenodo record; its API reports CC BY 4.0 and a 3.9 GB archive. | Later 8-channel continuous-speech and cross-corpus test. |
| [Somach two-configuration data](https://github.com/Somach-Systems-Inc/semg-silent-speech) | Public Git repository; its `LICENSE-DATA` says CC BY 4.0; about 87 MB CSV data from one participant and two configurations. | Exploratory configuration-transfer test, with session/time confounding made explicit. |

Keep raw data outside Git and preserve the publisher's attribution and license.
Do not treat dataset rights as rights to the publisher's hardware design or to
third-party patents.

## First completed offline pilot

`scripts/download_silentwear_subset.py` pins the SilentWear dataset revision,
checks its publisher-provided SHA-256 for each file, caps the planned download,
and emits an external manifest. S01–S03 silent recordings were imported: five
batches from each of three sessions for each participant. Each file is about
58 MB. The manifests and raw files live outside this repository at
`F:/Datasets/OpenAlterEgo/SilentWear` on the development machine.

From `software/python`, reproduce the import and exploratory run with an
external dataset directory:

```bash
python scripts/download_silentwear_subset.py --out /data/SilentWear --subject S01 --max-gb 1
python scripts/download_silentwear_subset.py --out /data/SilentWear --subject S02 --max-gb 1
python scripts/download_silentwear_subset.py --out /data/SilentWear --subject S03 --max-gb 1
uv run --frozen --extra public-data python scripts/silentwear_cohort.py \
  --root /data/SilentWear --subjects S01 S02 S03 \
  --output /runs/silentwear-development-cohort.json
```

The three imports total about 2.61 GB. `uv run` installs the project and
optional HDF5 reader. The result path must be new because the writer refuses
to replace an existing report.

`scripts/silentwear_cohort.py` uses session 1 for model fitting, session 2 for
selecting among five fixed six-channel subsets, and session 3 only for the
reported comparison. It extracts one centered 400-sample filtered-EMG window
per labeled event and fits a balanced logistic regression on log RMS and log
waveform length per channel. The benchmark reports balanced command/rest
accuracy, command accuracy, rest false activation, pairwise subset ranking,
and decision regret. A synthetic per-event channel feature shift probes
sensitivity; it is not calibrated electrode contact physics.

| Development comparison, S01–S03 | Mean pairwise ranking accuracy | Mean decision regret |
| --- | ---: | ---: |
| Original log features, choose by session-2 accuracy | 0.433 | 0.0246 |
| Label-free per-recording robust normalization, choose by session-2 accuracy | 0.667 | 0.0207 |

These are exploratory results from only three participants, a small fixed
candidate set, and a simple classifier. S01's regret increased from 0.0289 to
0.0516 after normalization. The full per-candidate output and input hashes are
in [the development result](results/silentwear-development-cohort.json).

## Next locked check

[The S04 holdout rule](02-public-holdout-s04.md) was frozen before importing
S04. [Its outcome](03-public-holdout-s04-outcome.md) improved ranking but missed
the declared regret threshold. Continue with larger public datasets, stronger signal
features, session-aware models, and physics perturbations calibrated against
observed session variation. Track each tested idea and its outcome; do not
relabel a repeatedly inspected participant as untouched.

The [two-configuration pilot](04-public-cross-configuration.md) is a separate
public-data stress test. It shows that simple feature-distribution adaptation
recovers some transfer accuracy but leaves a large gap to models trained on
the target configuration. It does not isolate a causal electrode effect.

The [simulator signature check](05-silentwear-simulator-signature.md) compares
both existing synthetic engines with an S01 training recording at the same
sample rate and channel count. It finds a large command/rest contrast and
spectral mismatch, providing a measurable calibration target before any claim
about wearable device physics.

The [expressive silent-voice track](06-expressive-silent-voice.md) runs in
parallel on public paired EMG/audio. It tests timing and prosodic cues where
acoustic targets exist, while keeping generated voice identity and optional
emotion controls separate from what silent EMG can establish.
