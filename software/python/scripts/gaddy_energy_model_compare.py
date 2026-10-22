"""Development-only comparison of causal EMG features for acoustic energy.

All source sessions supplied here have already been examined. Reserve a new
session for a protocol locked before its extraction or evaluation.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from scripts.gaddy_prosody_probe import (causal_filter_emg, decode_flac,
                                            paired_bootstrap_interval, trial_frames)
except ModuleNotFoundError:
    from gaddy_prosody_probe import (causal_filter_emg, decode_flac,
                                    paired_bootstrap_interval, trial_frames)


def lag_features(features: np.ndarray, lags: tuple[int, ...] = (0, 1, 2, 4)) -> np.ndarray:
    """Past frames only, with the first frame repeated inside each utterance."""
    indices = np.arange(len(features))
    return np.concatenate([features[np.maximum(indices - lag, 0)] for lag in lags], axis=1)


def activity_features(features: np.ndarray) -> np.ndarray:
    """Two global EMG activity summaries, without electrode-specific patterns."""
    return np.column_stack((features[:, :8].mean(axis=1), features[:, 8:].mean(axis=1)))


def shift_past(features: np.ndarray, frames: int = 20) -> np.ndarray:
    """Misalignment control, retaining only older features at each frame."""
    return features[np.maximum(np.arange(len(features)) - frames, 0)]


def session_rows(path: Path) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema"] != "openalterego.gaddy-voiced-subset.v1":
        raise ValueError(f"Unexpected manifest: {path}")
    ids = sorted({int(name.split("_", 1)[0]) for name in manifest["files"]})
    rows = []
    for trial_id in ids:
        info = json.loads((path / f"{trial_id}_info.json").read_text(encoding="utf-8"))
        if info["sentence_index"] < 0:
            continue
        emg = np.load(path / f"{trial_id}_emg.npy", allow_pickle=False).astype(np.float64)
        if emg.ndim != 2 or emg.shape[1] != 8 or not np.isfinite(emg).all():
            raise ValueError(f"Invalid EMG: {path}, {trial_id}")
        features, target, pitch = trial_frames(
            causal_filter_emg(emg), decode_flac(path / f"{trial_id}_audio_clean.flac")
        )
        if len(target):
            rows.append((features, target, pitch))
    return rows


def interior_voiced(pitch: np.ndarray) -> np.ndarray:
    """Oracle evaluation mask; never supplied to the prediction models."""
    voiced = pitch > 0
    interior = voiced.copy()
    for lag in (1, 2):
        interior[lag:] &= voiced[:-lag]
        interior[:-lag] &= voiced[lag:]
    return interior


def metrics(rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
            predictions: list[np.ndarray]) -> dict:
    targets = [row[1] for row in rows]
    masks = [interior_voiced(row[2]) for row in rows]
    all_error = np.concatenate([np.abs(y - p) for y, p in zip(targets, predictions)])
    voiced_error = np.concatenate([np.abs(y[m] - p[m]) for y, p, m in zip(targets, predictions, masks)])
    # Adjacent interior voiced-frame changes test contour dynamics separately
    # from utterance energy offset or easy silence detection.
    delta_target, delta_pred = [], []
    for y, p, mask in zip(targets, predictions, masks):
        adjacent = mask[1:] & mask[:-1]
        delta_target.append(np.diff(y)[adjacent])
        delta_pred.append(np.diff(p)[adjacent])
    dy, dp = np.concatenate(delta_target), np.concatenate(delta_pred)
    return {"all_frame_mae": float(all_error.mean()), "voiced_interior_mae": float(voiced_error.mean()),
            "voiced_delta_mae": float(np.mean(np.abs(dy - dp))),
            "voiced_delta_zero_mae": float(np.mean(np.abs(dy))),
            "frames": int(len(all_error)), "interior_frames": int(len(voiced_error)),
            "interior_deltas": int(len(dy))}


def paired_advantage(rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
                     selected: list[np.ndarray], control: list[np.ndarray]) -> dict:
    differences = np.array([
        np.mean(np.abs(row[1] - other)) - np.mean(np.abs(row[1] - chosen))
        for row, chosen, other in zip(rows, selected, control)
    ])
    return {"utterances": len(differences), "mean": float(differences.mean()),
            "ci95": paired_bootstrap_interval(differences)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.dev.resolve() in {path.resolve() for path in args.train}:
        parser.error("development session must differ from training sessions")
    train = [row for path in args.train for row in session_rows(path)]
    dev = session_rows(args.dev)
    if not train or not dev:
        parser.error("empty train or development session")
    train_y = np.concatenate([row[1] for row in train])
    train_current = np.concatenate([row[0] for row in train])
    train_lagged = np.concatenate([lag_features(row[0]) for row in train])
    train_activity = np.concatenate([lag_features(activity_features(row[0])) for row in train])
    dev_current = [row[0] for row in dev]
    dev_lagged = [lag_features(row[0]) for row in dev]
    dev_activity = [lag_features(activity_features(row[0])) for row in dev]
    current_model = make_pipeline(StandardScaler(), Ridge(alpha=100.0)).fit(train_current, train_y)
    candidates = {
        "mean_prior": [np.full(len(row[1]), train_y.mean()) for row in dev],
        "current_ridge": [current_model.predict(x) for x in dev_current],
    }
    # Predict once per candidate, then split by utterance to preserve boundaries.
    models = {
        "past_window_ridge": make_pipeline(StandardScaler(), Ridge(alpha=100.0)),
        "past_window_hgb": HistGradientBoostingRegressor(max_iter=120, max_leaf_nodes=15,
                                                          learning_rate=0.06, l2_regularization=10,
                                                          random_state=42),
    }
    for name, model in models.items():
        model.fit(train_lagged, train_y)
        candidates[name] = [model.predict(x) for x in dev_lagged]
        if name == "past_window_hgb":
            candidates["past_window_hgb_shift_1s"] = [model.predict(shift_past(x)) for x in dev_lagged]
    activity_model = HistGradientBoostingRegressor(max_iter=120, max_leaf_nodes=15,
                                                    learning_rate=0.06, l2_regularization=10,
                                                    random_state=42).fit(train_activity, train_y)
    candidates["global_activity_hgb"] = [activity_model.predict(x) for x in dev_activity]
    result = {
        "schema": "openalterego.gaddy-energy-model-compare.v1",
        "train_sessions": [path.name for path in args.train], "dev_session": args.dev.name,
        "preprocessing": "per-trial forward 60-Hz harmonic notches and 2-Hz highpass",
        "frame_step_ms": 50, "emg_window_ms": 100, "audio_window_ms": 40,
        "past_lags_frames": [0, 1, 2, 4],
        "oracle_mask_note": "Voiced interior mask is derived from evaluation audio pitch only; never model input.",
        "metrics": {name: metrics(dev, prediction) for name, prediction in candidates.items()},
        "hgb_all_frame_paired_advantage": {
            name: paired_advantage(dev, candidates["past_window_hgb"], prediction)
            for name, prediction in candidates.items() if name != "past_window_hgb"
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
