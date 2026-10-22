# External impedance spectra as a circuit stress test

The [previous circuit sensitivity run](71-electrode-loading-circuit-and-sensitivity.md)
drew hypothetical contact impedances capped near 2 MΩ. The open
[TextrodeMisc record](https://zenodo.org/records/17691766) supplies measured
complex skin-contact impedance spectra for several dry-electrode materials.
Its measurements use a **three-electrode forearm** setup. They constrain a
stress range, but do not identify OpenAlterEgo's facial contact impedance.

The publisher permits reuse under CC BY 4.0. The raw files live outside Git
in `F:\Datasets\OpenAlterEgo\ElectrodeImpedance-Zenodo17691766`; the
[manifest](data-sources/zenodo-17691766-textrode-impedance.json) retains their
published MD5 checksums and attribution. The
[importer](../../software/python/openalterego/research/impedance_spectra.py)
checks those hashes, parses real and imaginary impedance, averages duplicate
frequency observations, and interpolates in log frequency. The
[measured-contact circuit](../../software/python/openalterego/research/electrode_circuit.py)
loads each contact using the complex impedance at the stated frequency.

At 100 Hz the seven selected MXene–Ecoflex 5%/7% traces span **56.4–229 kΩ**
in magnitude. Three separate textile/spandex traces span **5.87–20.2 MΩ**.
We keep those material families separate. One other file (`MX 7,5% 15mm
-01.txt`) has negative real impedance near 100 Hz, so it is not inserted
into a passive small-signal circuit. The remaining ungrouped back-side
trace is also outside these cohorts. The spectra are individual published
traces, not independent facial subjects or repeated wearable placements.

## Bounded search replay

The [runner](../../software/python/scripts/measured_impedance_sensitivity.py)
draws a 100 Hz complex impedance independently for each of 40 **synthetic**
contacts, with replacement from one material family. It retains the earlier
hypothetical face lead fields, source and artifact covariances, contact
geometry, and fixed six-channel/eight-contact budget. The input capacitance
is zero; amplifier input resistance is swept between 1 MΩ and 100 MΩ;
CMRR is assumed to be 100 dB; and common-mode variance is an arbitrary
100. Development and evaluation scenarios use separate fixed seeds. No
contact placement, facial anatomy, analog front end, or decoder was measured.

| Material family | Assumed input resistance | Ideal choice | Circuit-aware choice | Held mean proxy: ideal → circuit-aware | Held minimum: ideal → circuit-aware |
| --- | ---: | --- | --- | ---: | ---: |
| MXene–Ecoflex | 100 MΩ | 32–37 | 32–37 | 3.827 → 3.827 | 3.283 → 3.283 |
| MXene–Ecoflex | 1 MΩ | 32–37 | 32–37 | 3.447 → 3.447 | 2.920 → 2.920 |
| Textile/spandex | 100 MΩ | 32–37 | 32–37 | 3.457 → 3.457 | 2.821 → 2.821 |
| Textile/spandex | 1 MΩ | 32–37 | 22, 32–36 | 0.336 → 0.350 | 0.096 → 0.089 |

The values are dimensionless narrowband log-determinant search proxies,
**not** recognition accuracy or measured information rate. Exact scores,
complex impedances, source hashes, and seeds are in the
[MXene 100 MΩ](results/measured-impedance-mxene-100m-v1.json),
[MXene 1 MΩ](results/measured-impedance-mxene-1m-v1.json),
[textile 100 MΩ](results/measured-impedance-ta-100m-v1.json), and
[textile 1 MΩ](results/measured-impedance-ta-1m-v1.json) records.

The extreme 1 MΩ/textile condition changes the selected program, with a
small higher held mean and a lower held minimum. It therefore establishes
**ranking sensitivity to an externally measured impedance range**, not
robust improvement or a better wearable. At 100 MΩ the layout does not
change. The earlier synthetic impedance cap omitted the textile range;
that omission mattered to this stress test.

## Next discriminating measurements

Before trusting layout recommendations, obtain facial skin-contact spectra
for the intended electrode materials and analog front-end input/noise
characteristics. Repeated placements and paired differential recordings
must test whether the assumed spatial lead fields and contact assignments
predict actual design rankings. With only external forearm impedance and
synthetic facial fields, neither material family is a validated prior for
the target wearable. More elaborate finite-element simulation is justified
only if it improves held-out intervention predictions beyond a calibrated
lower-dimensional model.
