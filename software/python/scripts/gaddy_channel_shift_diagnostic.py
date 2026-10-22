"""Compare unlabeled early-session channel distributions in Gaddy EMG."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance

from scripts.gaddy_channel_search import load, manifest_identity


def channel_shift(training, calibration):
    train_q25, train_q75 = np.quantile(training, [0.25, 0.75], axis=0)
    scale = np.maximum(train_q75 - train_q25, 0.1)
    values = [wasserstein_distance(training[:, feature] / scale[feature],
                                   calibration[:, feature] / scale[feature])
              for feature in range(16)]
    return [float(np.mean([values[channel], values[channel + 8]]))
            for channel in range(8)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--sessions", type=Path, nargs="+", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--calibration-utterances", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    training = np.concatenate([row[0] for row in load(
        args.train, args.cache_dir)])
    result = {"schema": "openalterego.gaddy-channel-shift-diagnostic.v1",
              "scope": "Unlabeled marginal feature drift only; retrospective vocalized sessions, no measured electrode health",
              "train": manifest_identity(args.train),
              "calibration_utterances": args.calibration_utterances,
              "sessions": {}}
    for path in args.sessions:
        rows = load(path, args.cache_dir)
        if len(rows) <= args.calibration_utterances:
            parser.error("calibration consumes a session")
        calibration = np.concatenate([
            row[0] for row in rows[:args.calibration_utterances]])
        shifts = channel_shift(training, calibration)
        result["sessions"][path.name] = {
            "source": manifest_identity(path),
            "feature_wasserstein_by_channel": shifts,
            "descending_shift_channels": list(map(int, np.argsort(shifts)[::-1])),
        }
        print(path.name, result["sessions"][path.name][
            "descending_shift_channels"], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
