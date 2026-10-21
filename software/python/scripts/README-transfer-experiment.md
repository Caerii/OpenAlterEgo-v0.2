# Pinned transfer experiment

From `software/python`, with `uv` and **separately licensed** Gowda real and
generated Gowda-shaped simulation sessions already present:

```bash
python scripts/transfer_experiment.py \
  --sim /data/gowda_sim --real /data/gowda_real \
  --out /runs/transfer-001
```

This prepares `manifest.json` and checks the 31-channel, 5 kHz input metadata,
required files, and the train/validation/test trial ranges. Execution requires
370 train, 30 validation, and 99 or 100 test trials (the documented source has
499 total trials). It hashes the full
input files. No training is run. Inspect the manifest, then use a fresh output
directory and add `--execute` from a clean committed checkout to run training.
The runner uses `uv run --frozen`, writes generated checkpoints and merged data
under `/runs/transfer-001/work`, and captures stdout, stderr, exit code, and a
result hash. Keep the run directory immutable after completion.

The wrapper does not establish consent, data rights, scientific validity, or
acceptance. The real and simulated sessions must be supplied with their own
provenance and access control. A completed process is a computational result;
prospective wearable validation follows the measured protocol in
`docs/research/00-measured-wearable-program.md`.
The historical `sim_only` tag means the model trains on simulated labeled
events; its SPD feature basis is fitted from real training trials. The result
must not be reported as zero-shot transfer without real-data access.
