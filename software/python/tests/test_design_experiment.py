"""The preregistration contract must reject post-hoc edits and incomplete outcomes."""

import copy
from datetime import datetime, timedelta, timezone

import pytest

from openalterego.research.design_experiment import register, score


@pytest.fixture
def protocol():
    return {
        "experiment_id": "OA-DESIGN-001",
        "hypothesis": "Candidate B improves next-session command accuracy after reseating.",
        "protocol_version": "v1",
        "model_hash": "sha256:model",
        "hardware_spec_hash": "sha256:hardware",
        "fit_data_hash": "sha256:training",
        "split_hash": "sha256:split",
        "baseline_id": "A",
        "baseline_method": "Empirical electrode selection on training sessions only.",
        "human_protocol_ref": "approved-study-001",
        "reviewer_ref": "independent-reviewer",
        "budget_usd": 100,
        "measurement_design": {
            "randomization": "Randomized ABBA or BAAB within participant.",
            "reseating": "Remove and reseat between blocks.",
            "placement_precision": "Measure pair centers to 1 mm.",
            "contact_measurement": "Record impedance at each block.",
            "future_session_holdout": "Second session on a later day is sealed.",
            "blinding": "Analyst sees candidate IDs only until predictions freeze.",
        },
        "candidates": [
            {"id": "A", "design_hash": "sha256:A", "predicted_accuracy": 0.70,
             "interval_low": 0.60, "interval_high": 0.80},
            {"id": "B", "design_hash": "sha256:B", "predicted_accuracy": 0.80,
             "interval_low": 0.70, "interval_high": 0.90},
        ],
    }


def test_register_exclusive_and_score_decision(tmp_path, protocol):
    path = tmp_path / "prereg.json"
    prereg = register(protocol, path)
    with pytest.raises(FileExistsError):
        register(protocol, path)
    outcome = {
        "protocol_sha256": prereg["protocol_sha256"],
        "measurement_manifest_hash": "sha256:heldout",
        "unblinded_utc": (datetime.fromisoformat(prereg["registered_utc"]) + timedelta(days=1)).isoformat(),
        "held_out_accuracy": {"A": 0.72, "B": 0.75},
    }
    result = score(prereg, outcome)
    assert result["decision_regret"] == pytest.approx(0)
    assert result["baseline_regret"] == pytest.approx(0.03)
    assert result["pairwise_ranking_accuracy"] == 1
    assert result["interval_coverage"] == 1


def test_score_rejects_changed_prediction_or_premature_outcome(tmp_path, protocol):
    prereg = register(protocol, tmp_path / "prereg.json")
    outcome = {
        "protocol_sha256": prereg["protocol_sha256"],
        "measurement_manifest_hash": "sha256:heldout",
        "unblinded_utc": datetime.now(timezone.utc).isoformat(),
        "held_out_accuracy": {"A": 0.8, "B": 0.7},
    }
    tampered = copy.deepcopy(prereg)
    tampered["protocol"]["candidates"][1]["predicted_accuracy"] = 0.9
    with pytest.raises(ValueError, match="hash mismatch"):
        score(tampered, outcome)
    outcome["unblinded_utc"] = (datetime.fromisoformat(prereg["registered_utc"]) - timedelta(seconds=1)).isoformat()
    with pytest.raises(ValueError, match="after registration"):
        score(prereg, outcome)


def test_scoring_counts_regret_and_rejects_missing_candidate(tmp_path, protocol):
    prereg = register(protocol, tmp_path / "prereg.json")
    outcome = {
        "protocol_sha256": prereg["protocol_sha256"],
        "measurement_manifest_hash": "sha256:heldout",
        "unblinded_utc": (datetime.fromisoformat(prereg["registered_utc"]) + timedelta(seconds=1)).isoformat(),
        "held_out_accuracy": {"A": 0.81, "B": 0.69},
    }
    result = score(prereg, outcome)
    assert result["decision_regret"] == pytest.approx(0.12)
    assert result["pairwise_ranking_accuracy"] == 0
    assert result["interval_coverage"] == 0
    del outcome["held_out_accuracy"]["A"]
    with pytest.raises(ValueError, match="exactly"):
        score(prereg, outcome)
