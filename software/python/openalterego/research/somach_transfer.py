"""Exploratory two-channel cross-configuration transfer on public CSV recordings."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


LABELS = ("DOWN", "LEFT", "NOISE", "RIGHT", "SILENCE", "UP")


def features_from_csv(path: Path) -> tuple[np.ndarray, str]:
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if len(rows) < 64:
        raise ValueError("recording too short")
    labels = {row["Label"].upper() for row in rows}
    if len(labels) != 1 or next(iter(labels)) not in LABELS:
        raise ValueError("unexpected labels")
    try:
        signal = np.asarray([[float(row["CH1"]), float(row["CH2"])] for row in rows], dtype=np.float64)
    except (KeyError, ValueError) as exc:
        raise ValueError("invalid ADC samples") from exc
    if not np.isfinite(signal).all() or np.any((signal < 0) | (signal > 4095)):
        raise ValueError("out-of-range ADC samples")
    centered = signal - np.median(signal, axis=0, keepdims=True)
    rms = np.sqrt(np.mean(centered ** 2, axis=0))
    waveform_length = np.mean(np.abs(np.diff(centered, axis=0)), axis=0)
    spectrum = np.abs(np.fft.rfft(centered, axis=0)) ** 2 / len(centered)
    frequencies = np.fft.rfftfreq(len(centered), d=1 / 250)
    bands = [np.mean(spectrum[(frequencies >= low) & (frequencies < high)], axis=0) for low, high in ((10, 30), (30, 60), (60, 110))]
    result = np.stack([rms, waveform_length, *bands], axis=1)
    return np.log1p(result).reshape(-1).astype(np.float32), next(iter(labels))


def load_folder(folder: Path) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]]]:
    paths = sorted(folder.glob("*.csv"))
    if not paths:
        raise ValueError(f"no CSV recordings: {folder}")
    extracted = []
    skipped = []
    for path in paths:
        try:
            extracted.append(features_from_csv(path))
        except ValueError as exc:
            skipped.append({"file": path.name, "reason": str(exc)})
    if not extracted:
        raise ValueError(f"no usable CSV recordings: {folder}")
    return np.stack([item[0] for item in extracted]), np.asarray([item[1] for item in extracted]), skipped


def _fit(features: np.ndarray, labels: np.ndarray, seed: int):
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed))
    return model.fit(features, labels)


def _score(model, features: np.ndarray, labels: np.ndarray) -> float:
    return float(balanced_accuracy_score(labels, model.predict(features)))


def _match_moments(source: np.ndarray, target_unlabeled: np.ndarray, *, scale: bool) -> np.ndarray:
    """Map source features into target moments without target labels."""
    centered = source - source.mean(axis=0)
    if scale:
        source_scale = np.maximum(source.std(axis=0), 0.1)
        target_scale = np.maximum(target_unlabeled.std(axis=0), 0.1)
        centered = centered / source_scale * target_scale
    return centered + target_unlabeled.mean(axis=0)


def benchmark(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray], *, seed: int = 42) -> dict:
    a_x, a_y = a
    b_x, b_y = b
    if a_x.shape[1] != 10 or b_x.shape[1] != 10 or set(a_y) != set(LABELS) or set(b_y) != set(LABELS):
        raise ValueError("both configurations need ten features and all six classes")
    a_fit, a_hold = train_test_split(np.arange(len(a_y)), test_size=0.2, stratify=a_y, random_state=seed)
    b_fit, b_hold = train_test_split(np.arange(len(b_y)), test_size=0.2, stratify=b_y, random_state=seed)
    a_model = _fit(a_x[a_fit], a_y[a_fit], seed)
    b_model = _fit(b_x[b_fit], b_y[b_fit], seed)
    shifted_model = _fit(_match_moments(a_x[a_fit], b_x[b_fit], scale=False), a_y[a_fit], seed)
    affine_model = _fit(_match_moments(a_x[a_fit], b_x[b_fit], scale=True), a_y[a_fit], seed)
    calibration = {}
    for shots in (1, 3, 5, 10):
        selected = []
        for label in LABELS:
            candidates = b_fit[b_y[b_fit] == label]
            if len(candidates) < shots:
                raise ValueError("not enough calibration samples")
            selected.extend(candidates[:shots])
        selected = np.asarray(selected)
        combined = _fit(np.concatenate([a_x[a_fit], b_x[selected]]), np.concatenate([a_y[a_fit], b_y[selected]]), seed)
        calibration[str(shots)] = _score(combined, b_x[b_hold], b_y[b_hold])
    return {
        "schema": "openalterego.cross-configuration.v1",
        "seed": seed,
        "samples": {"configuration_a": len(a_y), "configuration_b": len(b_y), "b_test": len(b_hold)},
        "configuration_a_within_balanced_accuracy": _score(a_model, a_x[a_hold], a_y[a_hold]),
        "configuration_b_within_balanced_accuracy": _score(b_model, b_x[b_hold], b_y[b_hold]),
        "a_to_b_balanced_accuracy": _score(a_model, b_x[b_hold], b_y[b_hold]),
        "a_to_b_unlabeled_mean_shift_accuracy": _score(shifted_model, b_x[b_hold], b_y[b_hold]),
        "a_to_b_unlabeled_affine_match_accuracy": _score(affine_model, b_x[b_hold], b_y[b_hold]),
        "a_plus_b_calibration_by_shots_per_class": calibration,
        "scope": "Same participant, two configurations recorded at different times; observational domain shift, not causal electrode effect.",
    }
