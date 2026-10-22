"""Measure the observed 14-channel versus fixed six-channel information gap."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from openalterego.research.program_search import normalize_batches
from scripts.silentwear_program_search import subject_sessions


SUBSETS = {"fixed_six": (0, 2, 5, 8, 11, 13), "all_fourteen": tuple(range(14))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", default=["S01", "S02", "S03"])
    args = parser.parse_args()
    if not set(args.subjects).issubset({"S01", "S02", "S03"}):
        parser.error("S04 is excluded from development")
    if args.output.exists():
        parser.error("output exists")
    result = {"schema": "openalterego.silentwear-channel-ceiling.v1",
              "split": "fit sessions 1+2, evaluate session 3",
              "scope": "Previously exposed development sessions; all fourteen *recorded* channels are not an unobserved-position oracle.",
              "normalization": "recording_robust", "classifier": "standardized class-balanced logistic C=1",
              "subsets": {name: list(channels) for name, channels in SUBSETS.items()},
              "subjects": {}}
    for subject in args.subjects:
        load_session, provenance, manifest_hash = subject_sessions(
            args.root, subject, args.cache_dir)
        train = normalize_batches(load_session(1) + load_session(2), "recording_robust")
        test = normalize_batches(load_session(3), "recording_robust")
        rows = {}
        for name, channels in SUBSETS.items():
            columns = [column for channel in channels
                       for column in (2 * channel, 2 * channel + 1)]
            model = make_pipeline(StandardScaler(), LogisticRegression(
                C=1, max_iter=2000, class_weight="balanced", random_state=42))
            model.fit(train[0][:, columns], train[1])
            predicted = model.predict(test[0][:, columns])
            matrix = confusion_matrix(test[1], predicted, labels=np.arange(9))
            recall = np.diag(matrix) / np.maximum(matrix.sum(axis=1), 1)
            rows[name] = {
                "balanced_accuracy": float(balanced_accuracy_score(test[1], predicted)),
                "command_balanced_accuracy": float(recall[1:].mean()),
                "per_class_recall": recall.tolist(), "confusion_matrix": matrix.tolist()}
        result["subjects"][subject] = {"scores": rows,
                                       "input_manifest_sha256": manifest_hash,
                                       "inputs": provenance}
        print(subject, {name: row["balanced_accuracy"] for name, row in rows.items()}, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
