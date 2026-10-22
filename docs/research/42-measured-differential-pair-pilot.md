# Measured adjacent-pair program search on dense forearm EMG

This pilot moves one layer beyond the [synthetic differential search](39-differential-program-search-pilot.md).
The public [Hyser dataset](https://physionet.org/content/hd-semg/2.0.0/)
contains 256 synchronized forearm channels on four 8×8 grids, acquired on
two separate days per participant. Its [original paper](https://physionet.org/static/published-projects/hd-semg/1.0.0/manuscript_author_version.pdf)
describes a reference electrode at the olecranon. We digitally subtract
neighboring publisher-preprocessed channels. This gives 448 horizontal or
vertical adjacent pairs with actual measured trial waveforms. It does **not**
recreate the contact and amplifier noise of a physical differential front
end, and forearm gestures are not facial silent speech.

The [cache builder](../../software/python/scripts/hyser_differential_cache.py)
verifies publisher SHA-256 hashes, WFDB initial values and checksums, and
channel names before computing log RMS and log waveform length for every
pair. The 195–204 trials per session and 448-pair feature matrices are
stored outside Git. The cache metadata records each pair's two source
contacts and the source-record identity. The
[pilot](../../software/python/scripts/hyser_differential_program_pilot.py)
samples 1,000 feasible programs per person. Each program takes two adjacent
pairs sharing one contact on each grid: **eight differential AFE channels,
12 active contacts, four-grid coverage**. Day-1 three-fold ridge CV chooses
one; a fixed central two-pair wedge per grid is the direct comparator.
Both use identical day-1 training and day-2 testing.

| Person | Searched differential | Fixed differential | Search minus fixed | Day-1/day-2 rank correlation |
| --- | ---: | ---: | ---: | ---: |
| 01 | 0.414 | 0.505 | −0.091 | +0.25 |
| 02 | 0.186 | 0.089 | +0.097 | +0.02 |
| 03 | 0.388 | 0.403 | −0.015 | +0.21 |
| 04 | 0.520 | 0.127 | +0.393 | +0.07 |
| 05 | 0.108 | 0.123 | −0.015 | +0.05 |

The searched choice improves the fixed differential program for two of five
people, with a +0.074 mean paired difference. Subject 04 dominates that
average; the rank correlations for 02, 04 and 05 are near zero. The fixed
differential program is an illustrative center placement and is weak on
several people. The [existing eight-site monopolar fixed benchmark](32-hyser-design-ranking-audit.md)
scored 0.571, 0.242, 0.441, 0.598 and 0.515 on the same five people,
respectively—above all five searched differential scores. That reference
uses eight recorded channels and a shared reference, whereas the pair
program uses 12 physical contacts and eight differential channels; it is a
practical challenge baseline, not a controlled analog-hardware comparison.

The [subject-01 result](results/hyser-differential-program-s01-v1.json) and
[subjects-02–05 result](results/hyser-differential-program-s02-05-v1.json)
record exact programs, source identities, candidate medians and retrospective
oracles. All these people and day-2 recordings are development data. The
oracles cannot be achieved by a day-1-only selector and are included only
to measure ranking headroom.

This establishes a real-signal, contact-accounted search lane, but does not
show a useful differential wearable advantage. The next physics question is
whether a calibrated model can predict which *measured pair changes* retain
class information across sessions. A decision rule must beat the strong
recorded-site baseline and the same-cost fixed-pair control on new people
before it guides a facial hardware layout.
