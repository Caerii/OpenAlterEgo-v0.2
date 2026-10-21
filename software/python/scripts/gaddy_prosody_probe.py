"""Exploratory voiced EMG to acoustic energy and F0 regression probe.

Requires ffmpeg and a bounded subset from extract_gaddy_voiced_subset.py.
The trial split is deterministic; frames from one trial never cross splits.
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy import signal
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def decode_flac(path: Path) -> np.ndarray:
    process = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1",
                              "-ar", "16000", "pipe:1"], capture_output=True, check=True)
    return np.frombuffer(process.stdout, dtype="<f4").astype(np.float64)


def estimate_f0(audio: np.ndarray) -> tuple[float, float]:
    """Simple normalized autocorrelation baseline, not a production pitch tracker."""
    x = audio - np.mean(audio)
    rms = float(np.sqrt(np.mean(x * x)))
    if rms < 1e-4:
        return 0.0, 0.0
    corr = signal.correlate(x, x, mode="full", method="fft")[len(x) - 1:]
    lo, hi = int(16000 / 400), int(16000 / 80)
    lag = int(np.argmax(corr[lo:hi + 1]) + lo)
    confidence = float(corr[lag] / max(corr[0], 1e-12))
    return (16000.0 / lag if confidence >= 0.45 else 0.0), confidence


def trial_frames(emg: np.ndarray, audio: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    duration = min(len(emg) / 1000, len(audio) / 16000)
    features, energy, f0 = [], [], []
    for center in np.arange(0.15, duration - 0.15, 0.05):
        e = emg[int(center * 1000) - 50:int(center * 1000) + 50]
        a = audio[int(center * 16000) - 320:int(center * 16000) + 320]
        if e.shape != (100, 8) or len(a) != 640:
            continue
        features.append(np.r_[np.log1p(np.sqrt(np.mean(e * e, axis=0))),
                                np.log1p(np.mean(np.abs(np.diff(e, axis=0)), axis=0))])
        energy.append(float(np.log10(max(np.sqrt(np.mean(a * a)), 1e-6))))
        pitch, _ = estimate_f0(a)
        f0.append(pitch)
    return np.asarray(features), np.asarray(energy), np.asarray(f0)


def score(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray) -> dict:
    model = make_pipeline(StandardScaler(), Ridge(alpha=100.0)).fit(train_x, train_y)
    baseline = DummyRegressor(strategy="mean").fit(train_x, train_y)
    prediction, constant = model.predict(test_x), baseline.predict(test_x)
    return {"frames": int(len(test_y)), "emg_mae": float(mean_absolute_error(test_y, prediction)),
            "mean_prior_mae": float(mean_absolute_error(test_y, constant)),
            "emg_r2": float(r2_score(test_y, prediction)),
            "mean_prior_r2": float(r2_score(test_y, constant))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.subset / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema"] != "openalterego.gaddy-voiced-subset.v1":
        parser.error("unexpected subset manifest")
    ids = sorted({int(name.split("_", 1)[0]) for name in manifest["files"]})
    valid = [i for i in ids if json.loads((args.subset / f"{i}_info.json").read_text())["sentence_index"] >= 0]
    rng = np.random.default_rng(42)
    rng.shuffle(valid)
    cut = int(len(valid) * 0.8)
    train_ids, test_ids = valid[:cut], valid[cut:]
    rows = {}
    for i in valid:
        emg = np.load(args.subset / f"{i}_emg.npy", allow_pickle=False).astype(np.float64)
        if emg.ndim != 2 or emg.shape[1] != 8 or not np.isfinite(emg).all():
            raise ValueError(f"bad EMG shape or values for trial {i}")
        rows[i] = trial_frames(emg, decode_flac(args.subset / f"{i}_audio_clean.flac"))
    train_x = np.concatenate([rows[i][0] for i in train_ids])
    test_x = np.concatenate([rows[i][0] for i in test_ids])
    results = {}
    for name, column in (("log_audio_rms", 1), ("log_audio_rms_voiced", 1), ("f0_hz", 2)):
        train_y = np.concatenate([rows[i][column] for i in train_ids])
        test_y = np.concatenate([rows[i][column] for i in test_ids])
        if name in ("f0_hz", "log_audio_rms_voiced"):
            train_pitch = np.concatenate([rows[i][2] for i in train_ids])
            test_pitch = np.concatenate([rows[i][2] for i in test_ids])
            train_mask, test_mask = train_pitch > 0, test_pitch > 0
            train_x_target, test_x_target = train_x[train_mask], test_x[test_mask]
            train_y, test_y = train_y[train_mask], test_y[test_mask]
        else:
            train_x_target, test_x_target = train_x, test_x
        results[name] = score(train_x_target, train_y, test_x_target, test_y)
        shuffled = score(train_x_target[np.random.default_rng(43).permutation(len(train_y))],
                         train_y, test_x_target, test_y)
        results[name]["shuffled_training_emg_mae"] = shuffled["emg_mae"]
        results[name]["shuffled_training_emg_r2"] = shuffled["emg_r2"]
    report = {"schema": "openalterego.gaddy-prosody-probe.v1", "source_md5": manifest["source_md5"],
              "source_path": manifest["source_path"], "seed": 42, "train_trial_ids": sorted(train_ids),
              "test_trial_ids": sorted(test_ids), "results": results,
              "scope": "Exploratory one-session vocalized speech; autocorrelation F0 targets; no silent prosody validation."}
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
