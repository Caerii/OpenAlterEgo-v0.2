"""Offline channel-subset and gain-drift benchmark for public EMG sessions.

Subsetting recorded channels cannot estimate signals at unrecorded positions.
The gain perturbation is a sensitivity test, not calibrated device physics.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def robust_recording_normalize(features: np.ndarray) -> np.ndarray:
    """Use only unlabeled feature distribution within the available recording."""
    median = np.median(features, axis=0)
    q25, q75 = np.quantile(features, [0.25, 0.75], axis=0)
    scale = np.maximum(q75 - q25, 0.1)
    return ((features - median) / scale).astype(np.float32)


def load_silentwear_events(path: Path, *, window_samples: int = 400, normalization: str = "none") -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Read the publisher's fixed-format HDF5 and create one feature row per event."""
    import h5py  # Optional dependency, needed only for this public dataset.

    with h5py.File(path, "r") as source:
        group = source["emg"]
        channels = [value.decode("ascii") for value in group["block1_items"][:]]
        expected = [f"Ch_{index}_filt" for index in range(11)] + [f"Ch_{index}_filt" for index in (13, 14, 15)]
        if channels != expected or list(group["block2_items"][:]) != [b"Label_int"]:
            raise ValueError("unexpected SilentWear channel or label layout")
        signals = np.asarray(group["block1_values"][:], dtype=np.float32)
        labels = np.asarray(group["block2_values"][:, 0], dtype=np.int64)
    if signals.ndim != 2 or signals.shape != (len(labels), len(channels)):
        raise ValueError("signal and label shapes differ")
    if not np.isfinite(signals).all() or not set(np.unique(labels)).issubset(set(range(9))):
        raise ValueError("non-finite signals or unexpected labels")
    boundaries = np.r_[0, np.flatnonzero(np.diff(labels)) + 1, len(labels)]
    features = []
    event_labels = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        if stop - start < window_samples:
            continue
        center = (start + stop) // 2
        sample = signals[center - window_samples // 2:center - window_samples // 2 + window_samples]
        rms = np.sqrt(np.mean(sample.astype(np.float64) ** 2, axis=0))
        waveform_length = np.mean(np.abs(np.diff(sample, axis=0)), axis=0)
        features.append(np.stack((np.log1p(rms), np.log1p(waveform_length)), axis=1).reshape(-1))
        event_labels.append(labels[start])
    extracted = np.asarray(features, dtype=np.float32)
    if normalization == "recording_robust":
        extracted = robust_recording_normalize(extracted)
    elif normalization != "none":
        raise ValueError("normalization must be none or recording_robust")
    return extracted, np.asarray(event_labels), channels


def fixed_subsets(n_channels: int = 14, width: int = 6) -> dict[str, list[int]]:
    if n_channels != 14 or width != 6:
        raise ValueError("this pilot requires the publisher's 14 channels and six-channel designs")
    return {
        "low_index": list(range(0, 6)),
        "mid_index": list(range(4, 10)),
        "high_index": list(range(8, 14)),
        "spread": [0, 2, 5, 8, 11, 13],
        "alternate": [1, 3, 5, 7, 9, 11],
    }


def _columns(subset: list[int]) -> list[int]:
    return [column for channel in subset for column in (2 * channel, 2 * channel + 1)]


def _metrics(true: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    commands = true != 0
    rest = true == 0
    return {
        "balanced_accuracy": float(balanced_accuracy_score(true, predicted)),
        "command_accuracy": float(np.mean(predicted[commands] == true[commands])),
        "rest_false_activation": float(np.mean(predicted[rest] != 0)),
    }


def feature_drift(features: np.ndarray, *, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Apply an uncalibrated common amplitude-feature shift per channel."""
    if sigma < 0:
        raise ValueError("sigma must be nonnegative")
    perturbation = rng.normal(0, sigma, size=(len(features), features.shape[1] // 2, 1))
    return features + np.repeat(perturbation, 2, axis=2).reshape(features.shape)


def benchmark(
    train: tuple[np.ndarray, np.ndarray],
    validation: tuple[np.ndarray, np.ndarray],
    test: tuple[np.ndarray, np.ndarray],
    *,
    seed: int = 42,
    drift_sigma: float = 0.35,
    drift_draws: int = 20,
) -> dict:
    """Rank on validation, report the previously untouched session as test."""
    train_x, train_y = train
    val_x, val_y = validation
    test_x, test_y = test
    for name, features, labels in (("train", train_x, train_y), ("validation", val_x, val_y), ("test", test_x, test_y)):
        if features.ndim != 2 or features.shape[1] != 28 or features.shape[0] != len(labels):
            raise ValueError(f"{name} must contain 14 channels and matched labels")
        if set(np.unique(labels)) != set(range(9)):
            raise ValueError(f"{name} lacks one or more rest/command labels")
    if drift_draws < 1:
        raise ValueError("drift_draws must be positive")
    candidates = {}
    for name, subset in fixed_subsets().items():
        columns = _columns(subset)
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed))
        model.fit(train_x[:, columns], train_y)
        val_metrics = _metrics(val_y, model.predict(val_x[:, columns]))
        test_metrics = _metrics(test_y, model.predict(test_x[:, columns]))
        val_perturbed = []
        test_perturbed = []
        for draw in range(drift_draws):
            # Same deterministic perturbations across candidates, so differences reflect channels.
            val_rng = np.random.default_rng(seed + draw)
            test_rng = np.random.default_rng(seed + 10_000 + draw)
            val_perturbed.append(_metrics(val_y, model.predict(feature_drift(val_x, sigma=drift_sigma, rng=val_rng)[:, columns]))["balanced_accuracy"])
            test_perturbed.append(_metrics(test_y, model.predict(feature_drift(test_x, sigma=drift_sigma, rng=test_rng)[:, columns]))["balanced_accuracy"])
        candidates[name] = {
            "channels": subset,
            "validation": val_metrics,
            "test": test_metrics,
            "simulated_feature_drift_validation_mean": float(np.mean(val_perturbed)),
            "simulated_feature_drift_test_mean": float(np.mean(test_perturbed)),
        }
    oracle = max(value["test"]["balanced_accuracy"] for value in candidates.values())
    choices = {}
    for rule, metric in (("validation_accuracy", lambda item: item["validation"]["balanced_accuracy"]),
                         ("validation_drift_robustness", lambda item: item["simulated_feature_drift_validation_mean"])):
        chosen = max(candidates, key=lambda name: metric(candidates[name]))
        choices[rule] = {"candidate": chosen, "test_balanced_accuracy": candidates[chosen]["test"]["balanced_accuracy"],
                         "test_decision_regret": oracle - candidates[chosen]["test"]["balanced_accuracy"]}
    pairs = [(a, b) for i, a in enumerate(candidates) for b in list(candidates)[i + 1:]]
    decisive = [(a, b) for a, b in pairs if candidates[a]["validation"]["balanced_accuracy"] != candidates[b]["validation"]["balanced_accuracy"] and candidates[a]["test"]["balanced_accuracy"] != candidates[b]["test"]["balanced_accuracy"]]
    correct = sum((candidates[a]["validation"]["balanced_accuracy"] - candidates[b]["validation"]["balanced_accuracy"]) * (candidates[a]["test"]["balanced_accuracy"] - candidates[b]["test"]["balanced_accuracy"]) > 0 for a, b in decisive)
    return {
        "schema": "openalterego.virtual-montage.v1",
        "seed": seed,
        "feature_drift_sigma": drift_sigma,
        "feature_drift_draws": drift_draws,
        "event_counts": {"train": len(train_y), "validation": len(val_y), "test": len(test_y)},
        "candidates": candidates,
        "validation_choice": choices["validation_accuracy"]["candidate"],
        "test_oracle_balanced_accuracy": oracle,
        "test_decision_regret": choices["validation_accuracy"]["test_decision_regret"],
        "choices": choices,
        "pairwise_ranking_accuracy": correct / len(decisive) if decisive else None,
        "decisive_pairs": len(decisive),
        "scope": "Existing-channel subset and uncalibrated synthetic feature-drift sensitivity; no unrecorded electrode positions.",
    }


def write_report(path: Path, result: dict) -> None:
    with path.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
