"""Frozen versus few-shot prototype decoding on later new-session batches."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from openalterego.research.program_search import normalize_batches
from openalterego.research.prototype_calibration import (
    adapt_centroids, class_centroids, predict_nearest)
from scripts.silentwear_program_search import subject_sessions


SUBSETS = {"fixed_six": (0, 2, 5, 8, 11, 13), "all_fourteen": tuple(range(14))}
BUDGETS = (0, 1, 5, 20)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists")
    result = {"schema": "openalterego.silentwear-calibration-curve.v1",
              "split": "fit sessions 1+2; calibrate from session 3 batch 1; evaluate session 3 batches 2-5",
              "normalization": "recording_robust", "subsets": {k: list(v) for k, v in SUBSETS.items()},
              "budgets_per_class": BUDGETS, "sampling": "seed 42, one nested permutation per class",
              "prototype_rule": "train-only StandardScaler; Euclidean centroids; 5-example prior pseudocount",
              "scope": "Retrospective development, offline per-recording normalization, already-exposed session 3.",
              "subjects": {}}
    for subject in ("S01", "S02", "S03"):
        load_session, provenance, manifest_hash = subject_sessions(
            args.root, subject, args.cache_dir)
        prior_batches = load_session(1) + load_session(2)
        later_batches = load_session(3)
        prior = normalize_batches(prior_batches, "recording_robust")
        calibration = normalize_batches(later_batches[:1], "recording_robust")
        evaluation = normalize_batches(later_batches[1:], "recording_robust")
        rng = np.random.default_rng(42)
        per_class_indices = {label: rng.permutation(np.flatnonzero(
            calibration[1] == label)) for label in range(9)}
        if any(len(indices) < max(BUDGETS) for indices in per_class_indices.values()):
            raise ValueError("insufficient calibration events for a class")
        scores = {}
        for name, channels in SUBSETS.items():
            columns = [column for channel in channels
                       for column in (2 * channel, 2 * channel + 1)]
            scaler = StandardScaler().fit(prior[0][:, columns])
            x_prior = scaler.transform(prior[0][:, columns])
            x_cal = scaler.transform(calibration[0][:, columns])
            x_eval = scaler.transform(evaluation[0][:, columns])
            prototype = class_centroids(x_prior, prior[1])
            logistic = make_pipeline(StandardScaler(), LogisticRegression(
                C=1, max_iter=2000, class_weight="balanced", random_state=42))
            logistic.fit(prior[0][:, columns], prior[1])
            row = {"frozen_logistic_balanced_accuracy": float(balanced_accuracy_score(
                evaluation[1], logistic.predict(evaluation[0][:, columns]))),
                   "prototype_accuracy_by_examples_per_class": {}}
            for count in BUDGETS:
                selected = np.concatenate([per_class_indices[label][:count]
                                           for label in range(9)]).astype(int)
                updated = adapt_centroids(prototype, x_cal[selected], calibration[1][selected])
                predicted = predict_nearest(x_eval, updated)
                row["prototype_accuracy_by_examples_per_class"][str(count)] = float(
                    balanced_accuracy_score(evaluation[1], predicted))
            scores[name] = row
        result["subjects"][subject] = {
            "scores": scores, "event_counts": {"prior": len(prior[1]),
                "calibration_pool": len(calibration[1]), "evaluation": len(evaluation[1])},
            "input_manifest_sha256": manifest_hash, "inputs": provenance}
        print(subject, scores, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
