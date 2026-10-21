"""Run an existing-channel subset benchmark on three SilentWear sessions."""

import argparse
import json
from pathlib import Path

import numpy as np

from openalterego.research.virtual_montage import benchmark, load_silentwear_events, sha256_file, write_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train", "validation", "test"):
        parser.add_argument(f"--{name}", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-drift-sigma", type=float, default=0.35)
    parser.add_argument("--feature-drift-draws", type=int, default=20)
    parser.add_argument("--normalization", choices=("none", "recording_robust"), default="none")
    args = parser.parse_args()
    inputs = {name: getattr(args, name) for name in ("train", "validation", "test")}
    paths = [path.resolve() for group in inputs.values() for path in group]
    if len(set(paths)) != len(paths):
        parser.error("train, validation and test files must be distinct")
    loaded = {name: [load_silentwear_events(path, normalization=args.normalization) for path in group] for name, group in inputs.items()}
    if len({tuple(value[2]) for group in loaded.values() for value in group}) != 1:
        parser.error("channel layouts differ across sessions")
    result = benchmark(*[(np.concatenate([item[0] for item in loaded[name]]), np.concatenate([item[1] for item in loaded[name]])) for name in ("train", "validation", "test")], drift_sigma=args.feature_drift_sigma, drift_draws=args.feature_drift_draws)
    result["inputs"] = {name: [{"sha256": sha256_file(path), "filename": path.name} for path in group] for name, group in inputs.items()}
    result["channels"] = loaded["train"][0][2]
    result["normalization"] = args.normalization
    write_report(args.output, result)
    print(json.dumps({"validation_choice": result["validation_choice"], "test_decision_regret": result["test_decision_regret"], "pairwise_ranking_accuracy": result["pairwise_ranking_accuracy"]}, indent=2))


if __name__ == "__main__":
    main()
