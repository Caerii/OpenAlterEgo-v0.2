# First public-data simulator signature check

This is an exploratory diagnostic, not a claim that the wearable physics has
been validated. The input is SilentWear S01 silent session 1 batch 1, a training
recording. Its publisher SHA-256 is
`c18a7bd4d3b2517be7a3e97e973179d1696d316f8ab9b68fa12e60f4255b98f7`.
The publisher's [acquisition constant](https://github.com/pulp-bio/SilentWear/blob/efc330b5bfe5863216eab9742be041ed99036665/utils/I_data_preparation/experimental_config.py)
sets 500 Hz. We used its 14 filtered channels and extracted one centered
400-sample window per rest or command run. Synthetic probes use the same channel
count and sample rate, seed 1337, and 40 simulated seconds. The default
biophysical model uses no claimed SilentWear electrode geometry.

| Statistic | Recorded S01 training batch | Heuristic default | Biophysical default |
| --- | ---: | ---: | ---: |
| Rest/command windows | 161 / 160 | 20 / 19 | 20 / 19 |
| Command-to-rest centered RMS ratio | 4.283 | 1.174 | 0.998 |
| Median absolute channel correlation | 0.213 | 0.166 | 0.200 |
| 100–200 Hz fraction of 20–200 Hz power | 0.166 | 0.486 | 0.401 |
| Median centered RMS, in each source's units | 25.551 | 35.856 | 8.272 |

The unitless activity contrast and spectral balance show clear mismatch. The
absolute amplitudes are descriptive because the public recording's scale is
not yet tied to the simulator's microvolt convention. Correlation proximity
does not redeem the command/rest or spectral mismatch. Synthetic event count
is much smaller, so this is a screening comparison, not a population estimate.
No simulator parameter was fitted to this batch in this result.

Reproduce from `software/python` after downloading the pinned publisher file:

```bash
uv run --frozen --extra public-data python scripts/silentwear_sim_signature.py \
  --recording /data/SilentWear/S01/silent/sess_1_batch_1.h5 \
  --engine biophysical --seconds 40 --output /runs/biophysical.json
uv run --frozen --extra public-data python scripts/silentwear_sim_signature.py \
  --recording /data/SilentWear/S01/silent/sess_1_batch_1.h5 \
  --engine heuristic --seconds 40 --output /runs/heuristic.json
```

The [biophysical result](results/silentwear-biophysical-signature-s01.json) and
[heuristic result](results/silentwear-heuristic-signature-s01.json) contain the
full statistics and input hash. A useful next iteration is to fit only on
S01–S03 session-1 recordings, select on session 2, and test whether calibrated
signal statistics predict session-3 or configuration-transfer outcomes. S04 is
already consumed as a holdout and should remain outside this calibration loop.
Matching summary statistics alone cannot establish correct anatomy, electrode
physics, or the effect of moving an electrode to an unrecorded location.
