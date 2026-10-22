"""Compare same-fit near-session and future-session decoding on recorded EMG."""

import argparse
import json
from pathlib import Path

import numpy as np
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
    result = {"schema": "openalterego.silentwear-session-gap.v1",
              "split": "fit session 1 and session 2 batches 1-4; compare session 2 batch 5 and all session 3 batches",
              "normalization": "recording_robust", "classifier": "standardized class-balanced logistic C=1",
              "scope": "Retrospective development diagnostic; unequal test sizes and prior session 3 exposure.",
              "subjects": {}}
    for subject in ("S01", "S02", "S03"):
        load_session, provenance, manifest_hash = subject_sessions(
            args.root, subject, args.cache_dir)
        day1, day2, day3 = load_session(1), load_session(2), load_session(3)
        train = normalize_batches(day1 + day2[:4], "recording_robust")
        near = normalize_batches(day2[4:], "recording_robust")
        future = normalize_batches(day3, "recording_robust")
        scores = {}
        for name, channels in SUBSETS.items():
            columns = [column for channel in channels
                       for column in (2 * channel, 2 * channel + 1)]
            model = make_pipeline(StandardScaler(), LogisticRegression(
                C=1, max_iter=2000, class_weight="balanced", random_state=42))
            model.fit(train[0][:, columns], train[1])
            scores[name] = {
                "near_session_balanced_accuracy": float(balanced_accuracy_score(
                    near[1], model.predict(near[0][:, columns]))),
                "future_session_balanced_accuracy": float(balanced_accuracy_score(
                    future[1], model.predict(future[0][:, columns]))),
                "events": {"train": len(train[1]), "near": len(near[1]),
                           "future": len(future[1])}}
        result["subjects"][subject] = {"scores": scores,
                                       "input_manifest_sha256": manifest_hash,
                                       "inputs": provenance}
        print(subject, scores, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
