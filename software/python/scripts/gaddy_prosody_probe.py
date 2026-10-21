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


def official_offline_emg(raw: np.ndarray, before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Gaddy read_emg.py filter and resample, before feature extraction.

    This uses filtfilt and future neighboring trials, so it is an offline parity
    condition and must never be described as streaming preprocessing. The
    six-sample offset in that source applies to its raw-EMG model input, not
    to the hand-engineered feature path used by this probe.
    """
    x = np.concatenate([before, raw, after], axis=0)
    for harmonic in range(1, 8):
        b, a = signal.iirnotch(60 * harmonic, 30, 1000)
        x = signal.filtfilt(b, a, x, axis=0)
    b, a = signal.butter(3, 2, "highpass", fs=1000)
    x = signal.filtfilt(b, a, x, axis=0)
    x = x[len(before):len(before) + len(raw)]
    times = np.arange(len(x)) / 1000
    sample_times = np.arange(0, times[-1], 1 / 516.79)
    return np.stack([np.interp(sample_times, times, x[:, channel])
                     for channel in range(x.shape[1])], axis=1)


def causal_filter_emg(raw: np.ndarray) -> np.ndarray:
    """Per-trial forward-only filtering at 1000 Hz; no adjacent trial access."""
    x = raw.copy()
    for harmonic in range(1, 8):
        b, a = signal.iirnotch(60 * harmonic, 30, 1000)
        x = signal.lfilter(b, a, x, axis=0)
    b, a = signal.butter(3, 2, "highpass", fs=1000)
    return signal.lfilter(b, a, x, axis=0)


def trial_frames(emg: np.ndarray, audio: np.ndarray,
                 emg_sample_rate: float = 1000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    duration = min(len(emg) / emg_sample_rate, len(audio) / 16000)
    features, energy, f0 = [], [], []
    window = round(0.1 * emg_sample_rate)
    for center in np.arange(0.15, duration - 0.15, 0.05):
        start = int(center * emg_sample_rate) - window // 2
        e = emg[start:start + window]
        a = audio[int(center * 16000) - 320:int(center * 16000) + 320]
        if e.shape != (window, 8) or len(a) != 640:
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


def robust_session_normalize(features: np.ndarray) -> np.ndarray:
    """Label-free per-session transform; requires the full session offline."""
    median = np.median(features, axis=0)
    q25, q75 = np.quantile(features, [0.25, 0.75], axis=0)
    return (features - median) / np.maximum(q75 - q25, 0.1)


def normalize_from_calibration(features: np.ndarray, calibration: np.ndarray) -> np.ndarray:
    """Use only earlier, unlabeled calibration frames to set session gain."""
    median = np.median(calibration, axis=0)
    q25, q75 = np.quantile(calibration, [0.25, 0.75], axis=0)
    return (features - median) / np.maximum(q75 - q25, 0.1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--test-subset", type=Path, help="independent voiced session for cross-session evaluation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-trials", type=int, default=0,
                        help="reserve this many earliest test trials for unlabeled causal calibration")
    parser.add_argument("--emg-preprocessing", choices=("raw", "official-offline", "causal-filter"),
                        default="raw")
    args = parser.parse_args()
    manifest = json.loads((args.subset / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema"] != "openalterego.gaddy-voiced-subset.v1":
        parser.error("unexpected subset manifest")
    ids = sorted({int(name.split("_", 1)[0]) for name in manifest["files"]})
    valid = [i for i in ids if json.loads((args.subset / f"{i}_info.json").read_text())["sentence_index"] >= 0]
    test_manifest = None
    if args.test_subset:
        test_manifest = json.loads((args.test_subset / "manifest.json").read_text(encoding="utf-8"))
        if test_manifest["schema"] != manifest["schema"] or test_manifest["source_md5"] != manifest["source_md5"]:
            parser.error("test subset provenance differs")
        if test_manifest["source_path"] == manifest["source_path"]:
            parser.error("cross-session test must use a distinct source directory")
        train_ids = valid
        test_ids = sorted({int(name.split("_", 1)[0]) for name in test_manifest["files"]})
        test_ids = [i for i in test_ids if json.loads((args.test_subset / f"{i}_info.json").read_text())["sentence_index"] >= 0]
    else:
        rng = np.random.default_rng(42)
        rng.shuffle(valid)
        cut = int(len(valid) * 0.8)
        train_ids, test_ids = valid[:cut], valid[cut:]

    def load_rows(path: Path, trial_ids: list[int]) -> dict:
        rows = {}
        for i in trial_ids:
            emg = np.load(path / f"{i}_emg.npy", allow_pickle=False).astype(np.float64)
            if emg.ndim != 2 or emg.shape[1] != 8 or not np.isfinite(emg).all():
                raise ValueError(f"bad EMG shape or values for trial {i}")
            rate = 1000
            if args.emg_preprocessing == "official-offline":
                empty = np.empty((0, 8))
                before = (np.load(path / f"{i-1}_emg.npy", allow_pickle=False)
                          if (path / f"{i-1}_emg.npy").exists() else empty)
                after = (np.load(path / f"{i+1}_emg.npy", allow_pickle=False)
                         if (path / f"{i+1}_emg.npy").exists() else empty)
                emg = official_offline_emg(emg, before, after)
                rate = 516.79
            elif args.emg_preprocessing == "causal-filter":
                emg = causal_filter_emg(emg)
            rows[i] = trial_frames(emg, decode_flac(path / f"{i}_audio_clean.flac"), rate)
        return rows

    train_rows = load_rows(args.subset, train_ids)
    test_rows = load_rows(args.test_subset or args.subset, test_ids)
    if args.calibration_trials < 0 or args.calibration_trials >= len(test_ids):
        parser.error("calibration trials must leave at least one test trial")
    evaluation_ids = test_ids[args.calibration_trials:]
    train_x = np.concatenate([train_rows[i][0] for i in train_ids])
    test_x = np.concatenate([test_rows[i][0] for i in evaluation_ids])
    train_x_robust = robust_session_normalize(train_x)
    test_x_robust = robust_session_normalize(test_x)
    test_x_calibrated = None
    if args.calibration_trials:
        calibration = np.concatenate([test_rows[i][0] for i in test_ids[:args.calibration_trials]])
        test_x_calibrated = normalize_from_calibration(test_x, calibration)
    results = {}
    for name, column in (("log_audio_rms", 1), ("log_audio_rms_voiced", 1), ("f0_hz", 2)):
        train_y = np.concatenate([train_rows[i][column] for i in train_ids])
        test_y = np.concatenate([test_rows[i][column] for i in evaluation_ids])
        if name in ("f0_hz", "log_audio_rms_voiced"):
            train_pitch = np.concatenate([train_rows[i][2] for i in train_ids])
            test_pitch = np.concatenate([test_rows[i][2] for i in evaluation_ids])
            train_mask, test_mask = train_pitch > 0, test_pitch > 0
            train_x_target, test_x_target = train_x[train_mask], test_x[test_mask]
            train_x_robust_target, test_x_robust_target = train_x_robust[train_mask], test_x_robust[test_mask]
            test_x_calibrated_target = test_x_calibrated[test_mask] if test_x_calibrated is not None else None
            train_y, test_y = train_y[train_mask], test_y[test_mask]
        else:
            train_x_target, test_x_target = train_x, test_x
            train_x_robust_target, test_x_robust_target = train_x_robust, test_x_robust
            test_x_calibrated_target = test_x_calibrated
        results[name] = score(train_x_target, train_y, test_x_target, test_y)
        normalized = score(train_x_robust_target, train_y, test_x_robust_target, test_y)
        results[name]["session_robust_emg_mae"] = normalized["emg_mae"]
        results[name]["session_robust_emg_r2"] = normalized["emg_r2"]
        if test_x_calibrated_target is not None:
            causal = score(train_x_robust_target, train_y, test_x_calibrated_target, test_y)
            results[name]["calibration_only_emg_mae"] = causal["emg_mae"]
            results[name]["calibration_only_emg_r2"] = causal["emg_r2"]
            causal_shuffled = score(
                train_x_robust_target[np.random.default_rng(43).permutation(len(train_y))],
                train_y, test_x_calibrated_target, test_y)
            results[name]["calibration_shuffled_emg_mae"] = causal_shuffled["emg_mae"]
        normalized_shuffled = score(
            train_x_robust_target[np.random.default_rng(43).permutation(len(train_y))],
            train_y, test_x_robust_target, test_y)
        results[name]["normalized_shuffled_emg_mae"] = normalized_shuffled["emg_mae"]
        shuffled = score(train_x_target[np.random.default_rng(43).permutation(len(train_y))],
                         train_y, test_x_target, test_y)
        results[name]["shuffled_training_emg_mae"] = shuffled["emg_mae"]
        results[name]["shuffled_training_emg_r2"] = shuffled["emg_r2"]
    def prompts(path: Path, trial_ids: list[int]) -> set[tuple[str, int]]:
        return {(info["book"], info["sentence_index"])
                for i in trial_ids
                for info in [json.loads((path / f"{i}_info.json").read_text(encoding="utf-8"))]}

    prompt_overlap = len(prompts(args.subset, train_ids) &
                         prompts(args.test_subset or args.subset, evaluation_ids))
    report = {"schema": "openalterego.gaddy-prosody-probe.v1", "source_md5": manifest["source_md5"],
              "source_path": manifest["source_path"],
              "test_source_path": test_manifest["source_path"] if test_manifest else manifest["source_path"],
              "seed": 42, "train_trial_ids": sorted(train_ids),
              "test_trial_ids": sorted(evaluation_ids),
              "calibration_trial_ids": sorted(test_ids[:args.calibration_trials]), "results": results,
              "emg_preprocessing": args.emg_preprocessing,
              "train_test_prompt_overlap": prompt_overlap,
              "scope": ("Exploratory two-session vocalized transfer; inspect prompt overlap count; "
                        if args.test_subset else "Exploratory one-session vocalized speech; ")
                       + "autocorrelation F0 targets; no silent prosody validation."}
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
