"""Development probe separating vocalized speech activity and slow energy contour.

The activity label and evaluation masks use recorded audio. Neither is a
deployment-ready silent-speech target. Use only already opened sessions while
selecting the model; lock a new session before examining it.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.decomposition import MiniBatchNMF
from sklearn.metrics import average_precision_score, f1_score, mean_absolute_error, roc_auc_score

try:
    from scripts.gaddy_energy_model_compare import activity_features, lag_features, session_rows
    from scripts.gaddy_prosody_probe import paired_bootstrap_interval
except ModuleNotFoundError:
    from gaddy_energy_model_compare import activity_features, lag_features, session_rows
    from gaddy_prosody_probe import paired_bootstrap_interval


def past_smooth(values: np.ndarray, width: int = 5) -> np.ndarray:
    """Causal 250 ms average with first-frame padding inside an utterance."""
    indices = np.arange(len(values))
    return np.mean([values[np.maximum(indices - lag, 0)] for lag in range(width)], axis=0)


def past_smooth_probability(values: np.ndarray, width: int = 5) -> np.ndarray:
    return past_smooth(values, width)


def top_k_codes(codes: np.ndarray, k: int = 2) -> np.ndarray:
    """Retain the k largest nonnegative latent components per frame."""
    if k < 1 or k > codes.shape[1] or np.any(codes < 0):
        raise ValueError("invalid code shape or sparsity")
    keep = np.argpartition(codes, -k, axis=1)[:, -k:]
    sparse = np.zeros_like(codes)
    rows = np.arange(len(codes))[:, None]
    sparse[rows, keep] = codes[rows, keep]
    return sparse


def slow_contour(rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
                 predictions: list[np.ndarray], lag: int = 4,
                 activity_threshold: float = -2.0) -> dict:
    actual, predicted = [], []
    for row, guess in zip(rows, predictions):
        target = past_smooth(row[1])
        # Require all intervening acoustic frames to be speech-active.
        active = row[1] > activity_threshold
        interior = active.copy()
        for offset in range(1, lag + 1):
            interior[offset:] &= active[:-offset]
        valid = interior[lag:]
        actual.append((target[lag:] - target[:-lag])[valid])
        predicted.append((guess[lag:] - guess[:-lag])[valid])
    dy, dp = np.concatenate(actual), np.concatenate(predicted)
    return {"delta_200ms_mae": float(mean_absolute_error(dy, dp)),
            "zero_change_mae": float(np.mean(np.abs(dy))),
            "frames": int(len(dy)),
            "direction_accuracy": float(np.mean(np.sign(dy) == np.sign(dp)))}


def paired_contour_advantage(rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
                             selected: list[np.ndarray], control: list[np.ndarray],
                             activity_threshold: float = -2.0) -> dict:
    differences = []
    for row, chosen, other in zip(rows, selected, control):
        target = past_smooth(row[1])
        active = row[1] > activity_threshold
        valid = active[4:].copy()
        for offset in range(1, 5):
            valid &= active[4 - offset:len(active) - offset]
        if not valid.any():
            continue
        dy = (target[4:] - target[:-4])[valid]
        dp_chosen = (chosen[4:] - chosen[:-4])[valid]
        dp_other = (other[4:] - other[:-4])[valid]
        differences.append(float(np.mean(np.abs(dy - dp_other)) -
                                 np.mean(np.abs(dy - dp_chosen))))
    values = np.asarray(differences)
    return {"utterances": len(values), "control_minus_selected_mean": float(values.mean()),
            "ci95": paired_bootstrap_interval(values)}


def make_model(regression: bool):
    common = dict(max_iter=120, max_leaf_nodes=15, learning_rate=0.06,
                  l2_regularization=10, random_state=42)
    return HistGradientBoostingRegressor(**common) if regression else HistGradientBoostingClassifier(**common)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-backend", choices=("python", "rust"), default="python")
    parser.add_argument("--feature-cache-dir", type=Path,
                        help="provenance-keyed cache outside Git; raw-derived arrays remain local")
    parser.add_argument("--profile-output", type=Path,
                        help="write per-stage wall times separately from scientific results")
    args = parser.parse_args()
    if args.dev.resolve() in {path.resolve() for path in args.train}:
        parser.error("development session must differ from training sessions")
    started = time.perf_counter()
    timings = {}
    stage = started
    train = [row for path in args.train for row in session_rows(
        path, feature_backend=args.feature_backend, estimate_pitch=False,
        cache_dir=args.feature_cache_dir)]
    timings["train_feature_load_s"] = time.perf_counter() - stage
    stage = time.perf_counter()
    dev = session_rows(args.dev, feature_backend=args.feature_backend,
                       estimate_pitch=False, cache_dir=args.feature_cache_dir)
    timings["dev_feature_load_s"] = time.perf_counter() - stage
    if not train or not dev:
        parser.error("empty train or development session")
    threshold = -2.0
    stage = time.perf_counter()
    y = np.concatenate([past_smooth(row[1]) for row in train])
    activity = np.concatenate([row[1] > threshold for row in train])
    x_full = np.concatenate([lag_features(row[0]) for row in train])
    x_global = np.concatenate([lag_features(activity_features(row[0])) for row in train])
    dev_full = [lag_features(row[0]) for row in dev]
    dev_global = [lag_features(activity_features(row[0])) for row in dev]
    timings["design_matrices_s"] = time.perf_counter() - stage
    # Sparse nonnegative channel patterns are learned without audio labels.
    # This is an observational structural prior, not an identified muscle model.
    current = np.concatenate([row[0] for row in train])
    rng = np.random.default_rng(42)
    sample = rng.choice(len(current), size=min(20000, len(current)), replace=False)
    dictionary = MiniBatchNMF(n_components=8, init="nndsvda", batch_size=1024,
                              max_iter=100, transform_max_iter=80, alpha_W=0.01,
                              alpha_H=0.01, l1_ratio=1.0, random_state=42)
    stage = time.perf_counter()
    dictionary.fit(current[sample])
    timings["dictionary_fit_s"] = time.perf_counter() - stage
    stage = time.perf_counter()
    latent_train = [dictionary.transform(row[0]) for row in train]
    latent_dev = [dictionary.transform(row[0]) for row in dev]
    sparse_train = [top_k_codes(code) for code in latent_train]
    sparse_dev = [top_k_codes(code) for code in latent_dev]
    x_latent = np.concatenate([lag_features(code) for code in latent_train])
    dev_latent = [lag_features(code) for code in latent_dev]
    x_sparse = np.concatenate([lag_features(code) for code in sparse_train])
    dev_sparse = [lag_features(code) for code in sparse_dev]
    timings["dictionary_transform_and_design_s"] = time.perf_counter() - stage

    stage = time.perf_counter()
    activity_model = make_model(False).fit(x_full, activity)
    dev_probability = [activity_model.predict_proba(x)[:, 1] for x in dev_full]
    timings["activity_fit_and_predict_s"] = time.perf_counter() - stage
    models = {
        "full_all_frames": (x_full, dev_full, np.ones(len(y), dtype=bool)),
        "global_all_frames": (x_global, dev_global, np.ones(len(y), dtype=bool)),
        "full_speech_trained": (x_full, dev_full, activity),
        "global_speech_trained": (x_global, dev_global, activity),
        "latent_dense_speech_trained": (x_latent, dev_latent, activity),
        "sparse_speech_trained": (x_sparse, dev_sparse, activity),
    }
    predictions = {}
    for name, (training_x, development_x, mask) in models.items():
        stage = time.perf_counter()
        fitted = make_model(True).fit(training_x[mask], y[mask])
        predictions[name] = [fitted.predict(x) for x in development_x]
        timings[f"model_{name}_fit_and_predict_s"] = time.perf_counter() - stage
    stage = time.perf_counter()
    inactive_mean = float(y[~activity].mean())
    predictions["two_stage_hard_gate"] = [np.where(probability >= 0.5, speech, inactive_mean)
                                       for probability, speech in zip(dev_probability,
                                                                       predictions["full_speech_trained"])]
    predictions["two_stage_soft_gate"] = [smoothed * speech + (1 - smoothed) * inactive_mean
                                          for probability, speech in zip(dev_probability,
                                                                          predictions["full_speech_trained"])
                                          for smoothed in [past_smooth_probability(probability)]]
    predictions["latent_dense_soft_gate"] = [past_smooth_probability(probability) * speech
                                       + (1 - past_smooth_probability(probability)) * inactive_mean
                                       for probability, speech in zip(dev_probability,
                                                                       predictions["latent_dense_speech_trained"])]
    predictions["sparse_soft_gate"] = [past_smooth_probability(probability) * speech
                                       + (1 - past_smooth_probability(probability)) * inactive_mean
                                       for probability, speech in zip(dev_probability,
                                                                       predictions["sparse_speech_trained"])]
    dev_labels = np.concatenate([row[1] > threshold for row in dev])
    probabilities = np.concatenate(dev_probability)
    result = {
        "schema": "openalterego.gaddy-contour-activity-dev.v1",
        "train_sessions": [path.name for path in args.train], "dev_session": args.dev.name,
        "feature_backend": args.feature_backend,
        "acoustic_activity_threshold_log10_rms": threshold,
        "target_smoothing": "causal 5-frame mean of log10 RMS, 50 ms frame step",
        "sparse_dictionary": {"components": 8, "alpha_w": 0.01, "alpha_h": 0.01,
                              "active_codes_per_frame": 2,
                              "train_code_zero_fraction": float(np.mean(np.concatenate(sparse_train) < 1e-8)),
                              "components_matrix": dictionary.components_.tolist()},
        "activity": {"frames": int(len(dev_labels)), "positive_rate": float(dev_labels.mean()),
                     "average_precision": float(average_precision_score(dev_labels, probabilities)),
                     "roc_auc": float(roc_auc_score(dev_labels, probabilities)),
                     "f1_at_half": float(f1_score(dev_labels, probabilities >= 0.5))},
        "energy": {},
    }
    for name, guess in predictions.items():
        truth = np.concatenate([past_smooth(row[1]) for row in dev])
        flat_guess = np.concatenate(guess)
        result["energy"][name] = {
            "all_frame_smoothed_mae": float(mean_absolute_error(truth, flat_guess)),
            "oracle_active_smoothed_mae": float(mean_absolute_error(truth[dev_labels], flat_guess[dev_labels])),
            "slow_contour": slow_contour(dev, guess, activity_threshold=threshold),
        }
    result["paired_contour_advantage"] = {
        selected: {
            control: paired_contour_advantage(dev, predictions[selected],
                                               [np.full(len(row[1]), inactive_mean) for row in dev]
                                               if control == "zero_change" else predictions[control])
            for control in ("zero_change", "global_speech_trained", "two_stage_soft_gate")
            if control != selected
        }
        for selected in ("full_speech_trained", "two_stage_soft_gate",
                         "latent_dense_soft_gate", "sparse_soft_gate")
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    timings["metrics_and_result_write_s"] = time.perf_counter() - stage
    timings["total_s"] = time.perf_counter() - started
    if args.profile_output:
        args.profile_output.parent.mkdir(parents=True, exist_ok=True)
        args.profile_output.write_text(json.dumps({
            "schema": "openalterego.gaddy-contour-stage-profile.v1",
            "feature_backend": args.feature_backend,
            "cache_enabled": args.feature_cache_dir is not None,
            "train_sessions": [path.name for path in args.train],
            "dev_session": args.dev.name,
            "timings": timings,
        }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"activity": result["activity"], "energy": result["energy"]}, indent=2))


if __name__ == "__main__":
    main()
