"""Measure how well one labeled current-day batch trains a local decoder."""

import argparse
import json
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
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
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists")
    result = {"schema": "openalterego.silentwear-session-local.v1",
              "split": "fit only session 3 batch 1; evaluate session 3 batches 2-5",
              "normalization": "recording_robust", "classifier": "standardized class-balanced logistic C=1",
              "scope": "Retrospective calibration diagnostic; 321 labeled events in calibration batch, including 20 of each command.",
              "subjects": {}}
    for subject in ("S01", "S02", "S03"):
        load_session, provenance, manifest_hash = subject_sessions(
            args.root, subject, args.cache_dir)
        batches = load_session(3)
        train = normalize_batches(batches[:1], "recording_robust")
        test = normalize_batches(batches[1:], "recording_robust")
        scores = {}
        for name, channels in SUBSETS.items():
            columns = [column for channel in channels
                       for column in (2 * channel, 2 * channel + 1)]
            model = make_pipeline(StandardScaler(), LogisticRegression(
                C=1, max_iter=2000, class_weight="balanced", random_state=42))
            model.fit(train[0][:, columns], train[1])
            scores[name] = float(balanced_accuracy_score(
                test[1], model.predict(test[0][:, columns])))
        result["subjects"][subject] = {"scores": scores,
                                       "event_counts": {"train": len(train[1]),
                                                        "test": len(test[1])},
                                       "input_manifest_sha256": manifest_hash,
                                       "inputs": provenance}
        print(subject, scores, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
