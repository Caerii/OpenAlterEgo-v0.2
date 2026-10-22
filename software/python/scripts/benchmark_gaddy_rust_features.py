"""Check Rust/Python feature parity and time in-memory trial framing.

The Rust path returns EMG features and acoustic energy, omitting the pitch
diagnostic that the contour experiment does not use. FLAC decoding and
in-memory framing are timed separately; filtering is excluded.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from gaddy_prosody_probe import causal_filter_emg, decode_flac, trial_frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--wheel-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.wheel_dir))
    import openalterego_accel

    audio_paths = [args.subset / f"{trial_id}_audio_clean.flac" for trial_id in range(args.trials)]
    maximum_audio_error = 0.0
    for path in audio_paths:
        python_audio = decode_flac(path)
        rust_audio = openalterego_accel.decode_gaddy_flac(str(path))
        if python_audio.shape != rust_audio.shape:
            raise ValueError(f"FLAC decode shape mismatch: {path}")
        maximum_audio_error = max(maximum_audio_error,
                                  float(np.max(np.abs(python_audio - rust_audio))))
    if maximum_audio_error > 1e-9:
        raise ValueError(f"Rust FLAC decode parity exceeded 1e-9: {maximum_audio_error}")

    rows = []
    for trial_id in range(args.trials):
        emg = causal_filter_emg(np.load(args.subset / f"{trial_id}_emg.npy", allow_pickle=False)
                                .astype(np.float64))
        audio = decode_flac(args.subset / f"{trial_id}_audio_clean.flac")
        rows.append((emg, audio))
    maximum_feature_error, maximum_energy_error, frame_count = 0.0, 0.0, 0
    def rust_features(emg, audio):
        duration = min(len(emg) / 1000, len(audio) / 16000)
        centers = np.arange(0.15, duration - 0.15, 0.05)
        return openalterego_accel.extract_energy_features(emg, audio, 1000.0, centers)

    for emg, audio in rows:
        py_features, py_energy, _ = trial_frames(emg, audio)
        rust_x, rust_energy = rust_features(emg, audio)
        if py_features.shape != rust_x.shape or py_energy.shape != rust_energy.shape:
            raise ValueError(f"frame shape mismatch: {py_features.shape} vs {rust_x.shape}")
        maximum_feature_error = max(maximum_feature_error, float(np.max(np.abs(py_features - rust_x))))
        maximum_energy_error = max(maximum_energy_error, float(np.max(np.abs(py_energy - rust_energy))))
        frame_count += len(py_energy)
    if maximum_feature_error > 1e-9 or maximum_energy_error > 1e-9:
        raise ValueError(f"Rust feature parity exceeded 1e-9: {maximum_feature_error}, {maximum_energy_error}")

    def timed(function):
        samples = []
        for _ in range(args.repetitions):
            start = time.perf_counter()
            for emg, audio in rows:
                function(emg, audio)
            samples.append(time.perf_counter() - start)
        return samples

    def timed_decode(function):
        samples = []
        for _ in range(args.repetitions):
            start = time.perf_counter()
            for path in audio_paths:
                function(path)
            samples.append(time.perf_counter() - start)
        return samples

    python_decode_seconds = timed_decode(decode_flac)
    rust_decode_seconds = timed_decode(lambda path: openalterego_accel.decode_gaddy_flac(str(path)))
    python_seconds = timed(trial_frames)
    python_energy_only_seconds = timed(lambda emg, audio: trial_frames(emg, audio, estimate_pitch=False))
    rust_seconds = timed(rust_features)
    result = {"schema": "openalterego.gaddy-rust-feature-benchmark.v1",
              "trials": args.trials, "frames": frame_count, "repetitions": args.repetitions,
              "python_seconds": python_seconds, "rust_seconds": rust_seconds,
              "python_decode_seconds": python_decode_seconds,
              "rust_decode_seconds": rust_decode_seconds,
              "median_decode_speedup": float(np.median(python_decode_seconds) /
                                             np.median(rust_decode_seconds)),
              "python_energy_only_seconds": python_energy_only_seconds,
              "median_speedup": float(np.median(python_seconds) / np.median(rust_seconds)),
              "median_energy_only_speedup": float(np.median(python_energy_only_seconds) /
                                                  np.median(rust_seconds)),
              "max_feature_abs_error": maximum_feature_error,
              "max_energy_abs_error": maximum_energy_error,
              "max_audio_abs_error": maximum_audio_error,
              "scope": "separate FLAC decode and in-memory frame extraction timings; filtering and model fitting excluded"}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
