"""Predefined six-channel layouts trained on two days and scored on day three."""

import argparse
import json
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from openalterego.research.program_search import MontageProgram, normalize_batches
from openalterego.research.virtual_montage import fixed_subsets
from scripts.silentwear_program_search import subject_sessions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", default=["S01", "S02", "S03"])
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    if not set(args.subjects).issubset({"S01", "S02", "S03"}):
        parser.error("S04 remains sealed for this development benchmark")
    result = {
        "schema": "openalterego.fixed-montage-two-day.v1",
        "split": "fit sessions 1 and 2; report session 3",
        "scope": "Predefined recorded-channel baselines, no physical intervention.",
        "subjects": {},
    }
    for subject in args.subjects:
        load_session, provenance, manifest_hash = subject_sessions(
            args.root, subject, args.cache_dir)
        train_batches = load_session(1) + load_session(2)
        test_batches = load_session(3)
        scores = {}
        for norm in ("none", "recording_robust"):
            train_x, train_y = normalize_batches(train_batches, norm)
            test_x, test_y = normalize_batches(test_batches, norm)
            for name, channels in fixed_subsets().items():
                program = MontageProgram(tuple(channels), norm)
                model = make_pipeline(StandardScaler(), LogisticRegression(
                    C=1.0, max_iter=2000, class_weight="balanced",
                    random_state=42))
                model.fit(train_x[:, program.columns], train_y)
                scores[f"{name}:{norm}"] = float(balanced_accuracy_score(
                    test_y, model.predict(test_x[:, program.columns])))
        result["subjects"][subject] = {
            "scores": scores, "input_manifest_sha256": manifest_hash,
            "inputs": provenance,
        }
        print(subject, scores, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
