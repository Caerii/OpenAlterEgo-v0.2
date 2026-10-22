"""Compare fresh-fit speed and contour quality on training-session holdouts."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

try:
    from scripts.gaddy_contour_activity_probe import past_smooth, slow_contour
    from scripts.gaddy_energy_model_compare import lag_features, session_rows
except ModuleNotFoundError:
    from gaddy_contour_activity_probe import past_smooth, slow_contour
    from gaddy_energy_model_compare import lag_features, session_rows


SETTINGS = {
    "baseline": {"max_bins": 255, "max_iter": 120},
    "bins128": {"max_bins": 128, "max_iter": 120},
    "bins64": {"max_bins": 64, "max_iter": 120},
    "iter80": {"max_bins": 255, "max_iter": 80},
    "bins128_iter80": {"max_bins": 128, "max_iter": 80},
}


def evaluate(train, dev, settings: dict) -> dict:
    threshold = -2.0
    x = np.concatenate([lag_features(row[0]) for row in train])
    target = np.concatenate([past_smooth(row[1]) for row in train])
    active = np.concatenate([row[1] > threshold for row in train])
    dev_x = [lag_features(row[0]) for row in dev]
    common = dict(max_leaf_nodes=15, learning_rate=0.06,
                  l2_regularization=10, random_state=42, **settings)
    stage = time.perf_counter()
    activity_model = HistGradientBoostingClassifier(**common).fit(x, active)
    activity_fit_s = time.perf_counter() - stage
    stage = time.perf_counter()
    energy_model = HistGradientBoostingRegressor(**common).fit(x[active], target[active])
    energy_fit_s = time.perf_counter() - stage
    stage = time.perf_counter()
    probabilities = [activity_model.predict_proba(values)[:, 1] for values in dev_x]
    energy = [energy_model.predict(values) for values in dev_x]
    inactive_mean = float(target[~active].mean())
    soft = [past_smooth(probability) * prediction +
            (1 - past_smooth(probability)) * inactive_mean
            for probability, prediction in zip(probabilities, energy)]
    prediction_s = time.perf_counter() - stage
    contour = slow_contour(dev, soft, activity_threshold=threshold)
    truth = np.concatenate([past_smooth(row[1]) for row in dev])
    all_frame_mae = float(np.mean(np.abs(truth - np.concatenate(soft))))
    return {"activity_fit_s": activity_fit_s, "energy_fit_s": energy_fit_s,
            "prediction_s": prediction_s,
            "activity_iterations": int(activity_model.n_iter_),
            "energy_iterations": int(energy_model.n_iter_),
            "all_frame_mae": all_frame_mae, "slow_contour": contour}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=Path, nargs="+", required=True)
    parser.add_argument("--holdout", nargs="+", required=True,
                        help="session directory names, drawn only from --sessions")
    parser.add_argument("--feature-cache-dir", type=Path, required=True)
    parser.add_argument("--feature-backend", choices=("rust", "python"), default="rust")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.repeats < 1 or len(args.sessions) < 2 or
            len({path.name for path in args.sessions}) != len(args.sessions)):
        parser.error("requires at least two unique sessions and positive repeats")
    if not set(args.holdout).issubset({path.name for path in args.sessions}):
        parser.error("holdouts must be among training-session paths")
    rows = {path.name: session_rows(path, feature_backend=args.feature_backend,
                                   estimate_pitch=False, cache_dir=args.feature_cache_dir)
            for path in args.sessions}
    records = []
    for holdout in args.holdout:
        train = [row for name, session in rows.items() if name != holdout
                 for row in session]
        dev = rows[holdout]
        for repeat in range(args.repeats):
            order = list(SETTINGS)
            if repeat % 2:
                order.reverse()
            for name in order:
                settings = SETTINGS[name]
                records.append({"holdout": holdout, "repeat": repeat,
                                "candidate": name, "settings": settings,
                                **evaluate(train, dev, settings)})
    result = {"schema": "openalterego.gaddy-fresh-fit-benchmark.v1",
              "sessions": list(rows), "holdouts": args.holdout,
              "feature_backend": args.feature_backend, "repeats": args.repeats,
              "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(records), "output": str(args.output)}))


if __name__ == "__main__":
    main()
