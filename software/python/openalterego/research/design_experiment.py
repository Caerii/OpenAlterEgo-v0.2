"""Register design predictions before measurement, then score observed decisions.

The files are locally hash-linked, not tamper-proof. An external timestamped
artifact store or governed receipt is needed for independent preregistration.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value


def _probability(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1]")
    return float(value)


def validate_protocol(protocol: dict[str, Any]) -> None:
    for name in ("experiment_id", "hypothesis", "protocol_version", "model_hash",
                 "hardware_spec_hash", "fit_data_hash", "split_hash", "baseline_id",
                 "baseline_method",
                 "human_protocol_ref", "reviewer_ref"):
        _required_text(protocol.get(name), name)
    if not isinstance(protocol.get("budget_usd"), (int, float)) or isinstance(protocol["budget_usd"], bool) or not math.isfinite(protocol["budget_usd"]) or protocol["budget_usd"] <= 0:
        raise ValueError("budget_usd must be positive and finite")
    candidates = protocol.get("candidates")
    if not isinstance(candidates, list) or len(candidates) < 2:
        raise ValueError("at least two candidates are required")
    ids: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("candidate must be an object")
        candidate_id = _required_text(candidate.get("id"), "candidate.id")
        _required_text(candidate.get("design_hash"), "candidate.design_hash")
        if candidate_id in ids:
            raise ValueError("duplicate candidate id")
        ids.add(candidate_id)
        lo = _probability(candidate.get("interval_low"), "interval_low")
        mean = _probability(candidate.get("predicted_accuracy"), "predicted_accuracy")
        hi = _probability(candidate.get("interval_high"), "interval_high")
        if not lo <= mean <= hi:
            raise ValueError("prediction must lie inside its interval")
    if protocol["baseline_id"] not in ids:
        raise ValueError("baseline_id must name a candidate")
    design = protocol.get("measurement_design")
    if not isinstance(design, dict):
        raise ValueError("measurement_design must be an object")
    for name in ("randomization", "reseating", "placement_precision",
                 "contact_measurement", "future_session_holdout", "blinding"):
        _required_text(design.get(name), f"measurement_design.{name}")


def register(protocol: dict[str, Any], path: Path) -> dict[str, Any]:
    """Create a preregistration exclusively; never replace one."""
    validate_protocol(protocol)
    registered = {
        "schema": "openalterego.design-prereg.v1",
        "registered_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol,
        "protocol_sha256": _digest(protocol),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        json.dump(registered, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    return registered


def score(prereg: dict[str, Any], outcomes: dict[str, Any]) -> dict[str, Any]:
    """Score a frozen prediction against separately collected held-out outcomes."""
    if prereg.get("schema") != "openalterego.design-prereg.v1":
        raise ValueError("unknown preregistration schema")
    protocol = prereg["protocol"]
    validate_protocol(protocol)
    if prereg.get("protocol_sha256") != _digest(protocol):
        raise ValueError("preregistration protocol hash mismatch")
    if outcomes.get("protocol_sha256") != prereg["protocol_sha256"]:
        raise ValueError("outcome protocol hash mismatch")
    _required_text(outcomes.get("measurement_manifest_hash"), "measurement_manifest_hash")
    _required_text(outcomes.get("unblinded_utc"), "unblinded_utc")
    try:
        registered_at = datetime.fromisoformat(prereg["registered_utc"])
        unblinded_at = datetime.fromisoformat(outcomes["unblinded_utc"])
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError("timestamps must be ISO 8601") from exc
    if registered_at.tzinfo is None or unblinded_at.tzinfo is None or unblinded_at <= registered_at:
        raise ValueError("outcome must be unblinded after registration")
    observed = outcomes.get("held_out_accuracy")
    candidates = protocol["candidates"]
    ids = {candidate["id"] for candidate in candidates}
    if not isinstance(observed, dict) or set(observed) != ids:
        raise ValueError("outcomes must contain exactly the registered candidates")
    observed = {key: _probability(value, f"held_out_accuracy.{key}") for key, value in observed.items()}
    predicted = {candidate["id"]: candidate["predicted_accuracy"] for candidate in candidates}
    ordered_ids = sorted(ids)
    concordant = 0
    decisive = 0
    for i, left in enumerate(ordered_ids):
        for right in ordered_ids[i + 1:]:
            predicted_difference = predicted[left] - predicted[right]
            observed_difference = observed[left] - observed[right]
            if predicted_difference and observed_difference:
                decisive += 1
                concordant += predicted_difference * observed_difference > 0
    predicted_winner = max(ordered_ids, key=lambda key: predicted[key])
    oracle_accuracy = max(observed.values())
    baseline_id = protocol["baseline_id"]
    coverage = sum(candidate["interval_low"] <= observed[candidate["id"]] <= candidate["interval_high"] for candidate in candidates)
    return {
        "schema": "openalterego.design-score.v1",
        "experiment_id": protocol["experiment_id"],
        "protocol_sha256": prereg["protocol_sha256"],
        "measurement_manifest_hash": outcomes["measurement_manifest_hash"],
        "predicted_winner": predicted_winner,
        "observed_oracle_accuracy": oracle_accuracy,
        "predicted_choice_accuracy": observed[predicted_winner],
        "decision_regret": oracle_accuracy - observed[predicted_winner],
        "baseline_id": baseline_id,
        "baseline_regret": oracle_accuracy - observed[baseline_id],
        "pairwise_ranking_accuracy": concordant / decisive if decisive else None,
        "decisive_pairs": decisive,
        "interval_coverage": coverage / len(candidates),
        "candidate_count": len(candidates),
    }
