"""Group-held-out layout recommender check on the development cohort."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    digest = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    if digest != metadata["npz_sha256"]:
        raise ValueError("development matrix hash mismatch")
    with np.load(args.dataset, allow_pickle=False) as matrix:
        predictors = matrix["predictors"]
        outcomes = matrix["targets"]
        subjects = list(map(str, matrix["subjects"]))
    if subjects != metadata["subjects"] or predictors.shape[:2] != outcomes.shape:
        raise ValueError("development matrix metadata mismatch")
    rows = {}
    for index, subject in enumerate(subjects):
        donor = np.arange(len(subjects)) != index
        x = predictors[donor].reshape(-1, predictors.shape[-1])
        target = (outcomes[donor] - outcomes[donor, -1, None]).reshape(-1)
        scaler = StandardScaler().fit(x)
        models = {
            "ridge_alpha1000": (Ridge(alpha=1000).fit(scaler.transform(x), target),
                                scaler.transform(predictors[index])),
            "hist_gradient_80_leaf8": (
                HistGradientBoostingRegressor(
                    max_iter=80, max_leaf_nodes=8, min_samples_leaf=100,
                    l2_regularization=10, learning_rate=0.05,
                    early_stopping=False, random_state=42).fit(x, target),
                predictors[index]),
        }
        rows[subject] = {}
        for name, (model, test_x) in models.items():
            prediction = model.predict(test_x)
            best = int(np.argmax(prediction[:-1]))
            rows[subject][name] = {
                "chosen_index": best,
                "predicted_advantage_vs_fixed": float(prediction[best] - prediction[-1]),
                "actual_advantage_vs_fixed": float(outcomes[index, best] - outcomes[index, -1]),
                "spearman_predicted_vs_actual": float(spearmanr(prediction, outcomes[index]).statistic),
                "fixed_day_two": float(outcomes[index, -1])}
        print(subject, {name: round(row["actual_advantage_vs_fixed"], 3)
                        for name, row in rows[subject].items()}, flush=True)
    summary = {}
    for name in next(iter(rows.values())):
        advantages = np.asarray([rows[subject][name]["actual_advantage_vs_fixed"]
                                 for subject in subjects])
        predicted = np.asarray([rows[subject][name]["predicted_advantage_vs_fixed"]
                                for subject in subjects])
        summary[name] = {"mean_advantage": float(advantages.mean()),
                         "worst_advantage": float(advantages.min()),
                         "subjects_improved": int(np.sum(advantages > 0)),
                         "subjects_harmed": int(np.sum(advantages < 0)),
                         "recommendations_over_0p05_margin": int(np.sum(predicted > 0.05))}
    result = {"schema": "openalterego.hyser-meta-loo.v1",
              "scope": "Development-only leave-one-person-out model check; no untouched-cohort result",
              "dataset_sha256": digest, "subjects": subjects,
              "model_settings": {"ridge_alpha": 1000,
                                 "hist_gradient": {"max_iter": 80, "max_leaf_nodes": 8,
                                                   "min_samples_leaf": 100,
                                                   "l2_regularization": 10,
                                                   "learning_rate": 0.05}},
              "summary": summary, "results": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
