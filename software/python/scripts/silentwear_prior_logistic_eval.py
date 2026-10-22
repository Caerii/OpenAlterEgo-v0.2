"""Score frozen cross-subject prior choices with a two-day logistic model."""

import argparse
import hashlib
import json
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from openalterego.research.program_search import MontageProgram, normalize_batches
from scripts.silentwear_program_search import subject_sessions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--choices", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    choice_bytes = args.choices.read_bytes()
    choices = json.loads(choice_bytes)
    if (choices.get("schema") != "openalterego.cross-subject-prior.v1" or
            sorted(choices["subjects"]) != ["S01", "S02", "S03"]):
        parser.error("expected frozen S01–S03 cross-subject choices")
    result = {
        "schema": "openalterego.cross-subject-prior-logistic.v1",
        "choice_sha256": hashlib.sha256(choice_bytes).hexdigest(),
        "split": "choose from cross-subject proxy study; fit sessions 1 and 2; evaluate session 3",
        "scope": "Retrospective development evaluation; same subjects and sessions were explored before.",
        "subjects": {},
    }
    for subject in sorted(choices["subjects"]):
        load_session, provenance, manifest_hash = subject_sessions(
            args.root, subject, args.cache_dir)
        train_batches = load_session(1) + load_session(2)
        test_batches = load_session(3)
        normalized = {norm: (normalize_batches(train_batches, norm),
                             normalize_batches(test_batches, norm))
                      for norm in ("none", "recording_robust")}
        outcomes = {}
        for name, row in choices["subjects"][subject]["methods"].items():
            norm, channels = row["program"]
            program = MontageProgram(tuple(channels), norm)
            (train_x, train_y), (test_x, test_y) = normalized[norm]
            model = make_pipeline(StandardScaler(), LogisticRegression(
                C=1.0, max_iter=2000, class_weight="balanced",
                random_state=42))
            model.fit(train_x[:, program.columns], train_y)
            outcomes[name] = {
                "program": row["program"],
                "test_balanced_accuracy": float(balanced_accuracy_score(
                    test_y, model.predict(test_x[:, program.columns]))),
            }
        result["subjects"][subject] = {
            "outcomes": outcomes,
            "input_manifest_sha256": manifest_hash,
            "inputs": provenance,
        }
        print(subject, {name: row["test_balanced_accuracy"] for name, row
                        in outcomes.items()}, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
