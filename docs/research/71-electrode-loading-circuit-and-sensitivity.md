# Electrode loading as an explicit hardware-search scenario

The earlier contact-level search assumed an ideal amplifier that reads
contact voltage without loading it. That omits a known mechanism:
electrode-interface impedance mismatch and finite amplifier input
impedance convert body common-mode voltage into differential voltage.
[Analog Devices describes this effect](https://www.analog.com/en/resources/design-notes/how-to-add-wilsons-central-terminalright-leg-drive-functions.html).
Measured skin–electrode impedances also vary with frequency and contact
conditions; [Cameron et al.](https://pubmed.ncbi.nlm.nih.gov/38083655/)
fit frequency-domain interface models to individual electrodes.

The new [circuit module](../../software/python/openalterego/research/electrode_circuit.py)
uses a small-signal, per-contact series resistance plus parallel RC model:

```text
Z_e(f) = R_s + (1/R_p + j 2π f C_p)^−1
H_i(f) = [1 + Z_e,i(f) (1/R_in + j 2π f C_in)]^−1
```

The existing differential incidence matrix `B` now sees loaded contact
potentials through `B diag(H)`. A uniform common-mode source leaks through
`B H 1`; a declared finite amplifier CMRR adds a residual term.
The model propagates independent contact artifacts, common-mode
interference, and AFE noise to a complex Hermitian channel covariance.
The [cached search](../../software/python/openalterego/research/differential_search.py)
precomputes that covariance once per scenario and still scores candidate
submatrices in batches. No parameter is silently inferred from EMG.

Focused tests verify equal contacts cancel ideal common mode; mismatched
contacts leak it; increasing input resistance reduces leakage; higher
frequency changes RC loading; the narrowband covariance is Hermitian
positive definite; and a nearly unloaded circuit converges to the old
ideal model. These are circuit identities and software checks, not a
measurement of the proposed wearable.

## Synthetic sensitivity check

The [runner](../../software/python/scripts/electrode_circuit_sensitivity.py)
reuses the earlier **hypothetical** 40-contact/two-chain lead-field
generator. It adds 12 development and 24 separate evaluation draws of
unmeasured contact impedances. Both searches choose six recorded channels
under eight active contacts. The model uses 100 Hz, 100 dB assumed CMRR,
zero modeled input capacitance, and an arbitrary common-mode variance
of 100 relative to unit-scale source covariance. Those values are
stress inputs, **not** fitted data or specified component performance.

| Assumed input resistance | Ideal-selected channels | Circuit-selected channels | Their held circuit log-det proxy |
| ---: | --- | --- | ---: |
| 1 MΩ | 32–37 | 32–37 | 3.519 |
| 100 MΩ | 32–37 | 32–37 | 3.828 |
| 1 GΩ | 32–37 | 32–37 | 3.828 |

Full [1 MΩ](results/electrode-circuit-1m-sensitivity-v1.json),
[100 MΩ](results/electrode-circuit-100m-sensitivity-v1.json), and
[1 GΩ](results/electrode-circuit-1g-sensitivity-v1.json) records include
assumptions, seeds, choices, contact counts, held scenario scores, and
local timings. Search took roughly 0.02–0.03 seconds for about 1,210
candidate evaluations, excluding scenario construction. These are
narrowband log-determinant **proxies**, not measured EMG information
rates or decoder accuracies.

The unchanged choice is itself diagnostic: the hand-built source field
strongly rewards the short chin chain, so this generator does not make
impedance mismatch decisive for layout selection. It would be misleading
to report the synthetic advantage over the illustrative cheek-only
fixed program as wearable progress. The immediate scientific need is
to measure or obtain contact-impedance spectra, amplifier input and
common-mode behavior, and array-specific spatial signal statistics,
then test whether circuit-aware predictions rank actual held-out
montages better. Electrode electrochemistry, current-noise coupling,
motion, nonlinearity, and reseating remain outside this small-signal
layer.
