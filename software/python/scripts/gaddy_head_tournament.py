"""Fresh-fit contour-head tournament on training-session holdouts only."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

try:
    from scripts.gaddy_contour_activity_probe import make_model, past_smooth, slow_contour
    from scripts.gaddy_energy_model_compare import lag_features, session_rows
except ModuleNotFoundError:
    from gaddy_contour_activity_probe import make_model, past_smooth, slow_contour
    from gaddy_energy_model_compare import lag_features, session_rows


def candidates():
    return {
        "ridge_1": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "ridge_100": make_pipeline(StandardScaler(), Ridge(alpha=100.0)),
        "spline_ridge": make_pipeline(
            StandardScaler(), SplineTransformer(n_knots=4, degree=2,
                                                 include_bias=False),
            Ridge(alpha=100.0)),
        "rbf_ridge": make_pipeline(StandardScaler(),
                                   RBFSampler(gamma=0.01, n_components=256,
                                              random_state=42),
                                   Ridge(alpha=1.0)),
        "hgb_64": make_model(True, max_bins=64),
        "hgb_255": make_model(True),
    }


def evaluate_fold(train, dev) -> tuple[dict, list[dict]]:
    threshold = -2.0
    x = np.concatenate([lag_features(row[0]) for row in train])
    y = np.concatenate([past_smooth(row[1]) for row in train])
    active = np.concatenate([row[1] > threshold for row in train])
    speech_x, speech_y = x[active], y[active]
    dev_x = [lag_features(row[0]) for row in dev]
    stage_wall, stage_cpu = time.perf_counter(), time.process_time()
    activity_model = make_model(False).fit(x, active)
    probabilities = [activity_model.predict_proba(values)[:, 1] for values in dev_x]
    smoothed = [past_smooth(probability) for probability in probabilities]
    activity_cost = {"wall_s": time.perf_counter() - stage_wall,
                     "cpu_s": time.process_time() - stage_cpu}
    inactive_mean = float(y[~active].mean())
    truth = np.concatenate([past_smooth(row[1]) for row in dev])
    records = []
    for name, model in candidates().items():
        stage_wall, stage_cpu = time.perf_counter(), time.process_time()
        model.fit(speech_x, speech_y)
        fit_wall = time.perf_counter() - stage_wall
        fit_cpu = time.process_time() - stage_cpu
        stage_wall, stage_cpu = time.perf_counter(), time.process_time()
        predictions = [model.predict(values) for values in dev_x]
        predict_wall = time.perf_counter() - stage_wall
        predict_cpu = time.process_time() - stage_cpu
        soft = [p * predicted + (1 - p) * inactive_mean
                for p, predicted in zip(smoothed, predictions)]
        records.append({"candidate": name,
                        "fit_wall_s": fit_wall, "fit_cpu_s": fit_cpu,
                        "predict_wall_s": predict_wall,
                        "predict_cpu_s": predict_cpu,
                        "all_frame_mae": float(np.mean(np.abs(truth - np.concatenate(soft)))),
                        "slow_contour": slow_contour(dev, soft,
                                                     activity_threshold=threshold)})
    return activity_cost, records


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
        cost, records = evaluate_fold(train, rows[holdout])
        folds.append({"holdout": holdout, "train_sessions": [name for name in rows
                                                              if name != holdout],
                      "activity_cost": cost, "records": records})
        print(f"completed {holdout}", flush=True)
    result = {"schema": "openalterego.gaddy-head-tournament.v1",
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
