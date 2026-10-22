"""Fresh-fit activity-gate tournament on training-session holdouts only."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from scripts.gaddy_contour_activity_probe import make_model, past_smooth, slow_contour
    from scripts.gaddy_energy_model_compare import lag_features, session_rows
except ModuleNotFoundError:
    from gaddy_contour_activity_probe import make_model, past_smooth, slow_contour
    from gaddy_energy_model_compare import lag_features, session_rows


def candidates():
    return {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, max_iter=500, random_state=42)),
        "sgd_logistic": make_pipeline(StandardScaler(), SGDClassifier(
            loss="log_loss", alpha=1e-4, max_iter=1000,
            tol=1e-3, random_state=42)),
        "hgb_64": make_model(False, max_bins=64),
        "hgb_255": make_model(False),
    }


def evaluate_fold(train, dev) -> dict:
    threshold = -2.0
    x = np.concatenate([lag_features(row[0]) for row in train])
    y = np.concatenate([past_smooth(row[1]) for row in train])
    active = np.concatenate([row[1] > threshold for row in train])
    dev_x = [lag_features(row[0]) for row in dev]
    dev_active = np.concatenate([row[1] > threshold for row in dev])
    truth = np.concatenate([past_smooth(row[1]) for row in dev])
    inactive_mean = float(y[~active].mean())
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    energy_model = make_model(True, max_bins=64).fit(x[active], y[active])
    energy_predictions = [energy_model.predict(values) for values in dev_x]
    energy_cost = {"wall_s": time.perf_counter() - start_wall,
                   "cpu_s": time.process_time() - start_cpu}
    records = []
    for name, model in candidates().items():
        start_wall, start_cpu = time.perf_counter(), time.process_time()
        model.fit(x, active)
        fit_wall = time.perf_counter() - start_wall
        fit_cpu = time.process_time() - start_cpu
        start_wall, start_cpu = time.perf_counter(), time.process_time()
        probabilities = [model.predict_proba(values)[:, 1] for values in dev_x]
        predict_wall = time.perf_counter() - start_wall
        predict_cpu = time.process_time() - start_cpu
        flat_probabilities = np.concatenate(probabilities)
        soft = [past_smooth(p) * predicted +
                (1 - past_smooth(p)) * inactive_mean
                for p, predicted in zip(probabilities, energy_predictions)]
        records.append({"candidate": name,
                        "fit_wall_s": fit_wall, "fit_cpu_s": fit_cpu,
                        "predict_wall_s": predict_wall,
                        "predict_cpu_s": predict_cpu,
                        "average_precision": float(average_precision_score(
                            dev_active, flat_probabilities)),
                        "roc_auc": float(roc_auc_score(dev_active,
                                                        flat_probabilities)),
                        "f1_at_half": float(f1_score(dev_active,
                                                     flat_probabilities >= 0.5)),
                        "all_frame_mae": float(np.mean(np.abs(
                            truth - np.concatenate(soft)))),
                        "slow_contour": slow_contour(dev, soft,
                                                     activity_threshold=threshold)})
    return {"energy_cost": energy_cost, "records": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=Path, nargs="+", required=True)
    parser.add_argument("--holdout", nargs="+", required=True)
    parser.add_argument("--feature-cache-dir", type=Path, required=True)
    parser.add_argument("--feature-backend", choices=("rust", "python"), default="rust")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    names = [path.name for path in args.sessions]
    if len(names) < 2 or len(set(names)) != len(names):
        parser.error("requires at least two uniquely named sessions")
    if not set(args.holdout).issubset(names):
        parser.error("holdouts must be named training sessions")
    rows = {path.name: session_rows(path, feature_backend=args.feature_backend,
                                   estimate_pitch=False, cache_dir=args.feature_cache_dir)
            for path in args.sessions}
    folds = []
    for holdout in args.holdout:
        train = [row for name, session in rows.items() if name != holdout
                 for row in session]
        folds.append({"holdout": holdout, "train_sessions": [name for name in rows
                                                              if name != holdout],
                      **evaluate_fold(train, rows[holdout])})
        print(f"completed {holdout}", flush=True)
    result = {"schema": "openalterego.gaddy-activity-tournament.v1",
              "sklearn_version": sklearn.__version__,
              "feature_backend": args.feature_backend,
              "feature_cache_enabled": True,
              "model_cache_enabled": False,
              "folds": folds}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "folds": len(folds)}))


if __name__ == "__main__":
    main()
