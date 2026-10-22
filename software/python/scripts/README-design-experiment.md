# Prospective design experiment

Run from `software/python` after freezing the analysis and approved human
measurement plan:

```bash
uv run --frozen python scripts/design_experiment.py register protocol.json prereg.json
```

The protocol JSON has this shape (illustrative predictions only):

```json
{
  "experiment_id": "OA-DESIGN-001",
  "hypothesis": "Layout B improves command accuracy after reseating.",
  "protocol_version": "approved-protocol-v1",
  "model_hash": "sha256:...",
  "hardware_spec_hash": "sha256:...",
  "fit_data_hash": "sha256:...",
  "split_hash": "sha256:...",
  "baseline_id": "A",
  "baseline_method": "Empirical selection using training sessions only",
  "human_protocol_ref": "approved-study-reference",
  "reviewer_ref": "independent-review-reference",
  "budget_usd": 100,
  "measurement_design": {
    "randomization": "Randomized ABBA or BAAB within participant",
    "reseating": "Remove and reseat between blocks",
    "placement_precision": "Measure pair centers to 1 mm",
    "contact_measurement": "Record contact impedance at each block",
    "future_session_holdout": "Later session remains sealed until prediction registration",
    "blinding": "Analyst sees candidate IDs until predictions freeze"
  },
  "candidates": [
    {"id": "A", "design_hash": "sha256:...", "predicted_accuracy": 0.70,
     "interval_low": 0.60, "interval_high": 0.80},
    {"id": "B", "design_hash": "sha256:...", "predicted_accuracy": 0.80,
     "interval_low": 0.70, "interval_high": 0.90}
  ]
}
```

The command creates `prereg.json` exclusively. Put its complete content and
hash into a timestamped immutable evidence store **before unblinding**. A
local file and self-reported timestamp alone do not prove this order.

The measurement team later supplies an outcome JSON with the exact
`protocol_sha256` printed at registration, a `measurement_manifest_hash`,
`unblinded_utc`, and `held_out_accuracy` for every candidate:

```json
{
  "protocol_sha256": "sha256-from-prereg-without-prefix",
  "measurement_manifest_hash": "sha256:...",
  "unblinded_utc": "2026-10-22T14:00:00+00:00",
  "held_out_accuracy": {"A": 0.72, "B": 0.75}
}
```

Then score once and retain the full outcome and review record:

```bash
uv run --frozen python scripts/design_experiment.py score prereg.json outcomes.json score.json
```

`pairwise_ranking_accuracy` excludes predicted or observed ties and is `null`
when no decisive pair exists. `decision_regret` is the best measured candidate
accuracy minus the measured accuracy of the predicted winner. These point
estimates need participant-level uncertainty and future-session replication
before a physical design claim. This utility does not conduct or approve a
human study, enforce blinding, or provide an independent timestamp.
