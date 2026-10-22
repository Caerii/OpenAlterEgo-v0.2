"""Test rest detection and command decoding as separate fixed classifiers."""

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


def model():
    return make_pipeline(StandardScaler(), LogisticRegression(
        C=1, max_iter=2000, class_weight="balanced", random_state=42))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists")
    result = {"schema": "openalterego.silentwear-hierarchical.v1",
              "split": "fit sessions 1+2; evaluate session 3",
              "normalization": "recording_robust", "classifiers": "C=1 balanced logistic, train-only scaler",
              "gate": "binary command probability > 0.5; then eight-class command model",
              "scope": "Retrospective development on previously exposed session 3; fixed recorded channels and isolated events.",
              "subjects": {}}
    for subject in ("S01", "S02", "S03"):
        load_session, provenance, manifest_hash = subject_sessions(
            args.root, subject, args.cache_dir)
        train = normalize_batches(load_session(1) + load_session(2), "recording_robust")
        test = normalize_batches(load_session(3), "recording_robust")
        scores = {}
        for name, channels in SUBSETS.items():
            columns = [column for channel in channels
                       for column in (2 * channel, 2 * channel + 1)]
            x, y = train[0][:, columns], train[1]
            xt, yt = test[0][:, columns], test[1]
            flat = model().fit(x, y)
            gate = model().fit(x, (y != 0).astype(int))
            commands = model().fit(x[y != 0], y[y != 0])
            flat_pred = flat.predict(xt)
            command_pred = commands.predict(xt)
            is_command = gate.predict_proba(xt)[:, 1] > 0.5
            hierarchical = np.where(is_command, command_pred, 0)
            confusion = confusion_matrix(yt, hierarchical, labels=np.arange(9))
            recall = np.diag(confusion) / np.maximum(confusion.sum(axis=1), 1)
            scores[name] = {
                "flat_balanced_accuracy": float(balanced_accuracy_score(yt, flat_pred)),
                "hierarchical_balanced_accuracy": float(balanced_accuracy_score(yt, hierarchical)),
                "gate_balanced_accuracy": float(balanced_accuracy_score(
                    yt != 0, is_command)),
                "command_conditional_balanced_accuracy": float(balanced_accuracy_score(
                    yt[yt != 0], command_pred[yt != 0])),
                "hierarchical_rest_recall": float(recall[0]),
                "hierarchical_command_mean_recall": float(recall[1:].mean())}
        result["subjects"][subject] = {"scores": scores,
                                       "input_manifest_sha256": manifest_hash,
                                       "inputs": provenance}
        print(subject, {name: {key: value for key, value in row.items()
                              if key.endswith("balanced_accuracy")}
                        for name, row in scores.items()}, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
