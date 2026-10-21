# Public two-configuration transfer pilot

Source: [Somach Systems' public dataset](https://github.com/Somach-Systems-Inc/semg-silent-speech)
at commit `5ad28f5bdca7e56610285ae9302f1890df5560e6`; CSV data license
CC BY 4.0, attributed to Carl Vincent Ladres Kho. The external clone is
outside OpenAlterEgo's Git checkout. The source describes one participant and
two 250 Hz, 12-bit, two-channel setups: chin plus under-chin (A) and chin plus
throat (B), recorded at different times. The comparison is observational.

`scripts/somach_transfer.py` extracts log RMS, waveform length, and three
spectral-band powers per channel from each recording. For each shared speech
phase it makes a fixed stratified 80/20 split within each configuration. It
compares within-configuration models, A-to-B transfer on the same B test
partition, label-free mean or affine moment matching using B training features,
and a few labeled B calibration examples per class. The code skips malformed
recordings and logs each exclusion. One short A overt recording was excluded.

| Balanced accuracy, mean across overt, mouthing, exaggerated phases | Score |
| --- | ---: |
| Within configuration A | 0.633 |
| Within configuration B | 0.661 |
| Train on A, test on B | 0.328 |
| Train on A after unlabeled B mean matching | 0.378 |
| Train on A after unlabeled B mean and scale matching | 0.406 |

For the silent-like *mouthing* phase alone, A-to-B transfer was 0.233 versus
0.550/0.583 within A/B. Affine matching raised transfer to 0.300. Chance for
six balanced classes is about 0.167. The [machine-readable result](results/somach-cross-configuration-phases.json)
includes each phase, the fixed seed, sample counts, adaptation scores, and the
skipped-file record.

The improvement from affine matching is a useful offline lead, not a proven
hardware effect. Unlabeled B features are available to the adaptation method;
the B test labels are not. Both configurations share a participant but differ
in time, preparation and possibly acquisition conditions. The three phase
results were explored after observing the initial mouthing result, so they
are development evidence. Use an independently sourced corpus or untouched
configuration/session split before making a general transfer claim.
