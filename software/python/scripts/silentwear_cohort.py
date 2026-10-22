"""Evaluate fixed virtual montages across public SilentWear participants and days."""

import argparse
import json
from pathlib import Path

import numpy as np

from openalterego.research.virtual_montage import (
    benchmark,
    load_silentwear_events,
    robust_recording_normalize,
    sha256_file,
    write_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = {}
    for subject in args.subjects:
        base = args.root / subject / "silent"
        sessions = {session: sorted(base.glob(f"sess_{session}_batch_*.h5")) for session in (1, 2, 3)}
        if any(len(paths) != 5 for paths in sessions.values()):
            parser.error(f"{subject}: expected five batches in each of three sessions")
        loaded = {session: [load_silentwear_events(path) for path in paths] for session, paths in sessions.items()}
        channels = {tuple(item[2]) for session in loaded.values() for item in session}
        if len(channels) != 1:
            parser.error(f"{subject}: channel layouts differ")
        reports[subject] = {"inputs": {f"session_{session}": [{"file": path.name, "sha256": sha256_file(path)} for path in paths] for session, paths in sessions.items()},
                            "normalizations": {}}
        for normalization in ("none", "recording_robust"):
            combined = []
            for session in (1, 2, 3):
                items = loaded[session]
                features = [robust_recording_normalize(item[0]) if normalization == "recording_robust" else item[0] for item in items]
                combined.append((np.concatenate(features), np.concatenate([item[1] for item in items])))
            reports[subject]["normalizations"][normalization] = benchmark(*combined)
            result = reports[subject]["normalizations"][normalization]
            print(subject, normalization, result["choices"], flush=True)
    summary = {}
    for normalization in ("none", "recording_robust"):
        summary[normalization] = {}
        for rule in ("validation_accuracy", "validation_drift_robustness"):
            regrets = [reports[subject]["normalizations"][normalization]["choices"][rule]["test_decision_regret"] for subject in args.subjects]
            summary[normalization][rule] = {"mean_test_decision_regret": float(np.mean(regrets)),
                                             "regret_by_subject": dict(zip(args.subjects, regrets))}
    write_report(args.output, {"schema": "openalterego.virtual-montage-cohort.v1", "subjects": reports,
                               "summary": summary, "split": "train session 1, candidate selection session 2, report session 3",
                               "scope": "Offline public data; virtual subsets of recorded channels only."})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
