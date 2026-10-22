"""Freeze facial six-channel programs before later-block query scoring."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from openalterego.research import dense_programs
from openalterego.research.dense_programs import (
    prepare_cv_folds, prepare_fold, score_programs)
from scripts import faceemg11_extract, faceemg11_program_search
from scripts.faceemg11_program_search import (
    CANDIDATES, FIXED, load_subject, split_subject)


DEVELOPMENT = tuple(f"{subject:02d}" for subject in range(1, 7))
METHOD_CHECK = tuple(f"{subject:02d}" for subject in range(7, 13))
CHOICE_NAMES = ("personal_train_cv", "personal_forward",
                "global_train_cv", "global_forward", "fixed_index_spread")
CODE_FILES = (Path(__file__), Path(dense_programs.__file__),
              Path(faceemg11_extract.__file__),
              Path(faceemg11_program_search.__file__))


def canonical_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def code_hash() -> str:
    return hashlib.sha256(b"".join(path.read_bytes() for path in CODE_FILES)).hexdigest()


def write_exclusive(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def selection_scores(feature_cache: Path, subject: str):
    features, labels, groups, metadata = load_subject(feature_cache, subject)
    train, validation, _ = split_subject(features, labels, groups)
    cv = score_programs(prepare_cv_folds(*train, features_per_site=2),
                        CANDIDATES, features_per_site=2)
    forward = score_programs(
        [prepare_fold(train, validation, features_per_site=2)],
        CANDIDATES, features_per_site=2)
    return cv, forward, metadata["member_sha256"]


def select(feature_cache: Path, output: Path):
    if output.exists():
        raise ValueError("selection already exists")
    result = {
        "schema": "openalterego.faceemg11-method-selection.v1",
        "source": "https://doi.org/10.5281/zenodo.22137268",
        "code_sha256": code_hash(),
        "development_subjects": DEVELOPMENT,
        "method_check_subjects": METHOD_CHECK,
        "candidate_count": len(CANDIDATES),
        "split": "blocks 1-15 fit/CV, 16-20 forward validation; score on blocks 21-30 only after selection is pushed",
        "primary": "personal train-CV selected minus global forward-validation selected, both six channels",
        "success_rule": "mean gain >=0.05, >=5/6 people improve, no loss below -0.05",
        "development_sources": {}, "global_choices": {},
        "method_choices": {},
    }
    total_cv = np.zeros(len(CANDIDATES), dtype=np.float64)
    total_forward = np.zeros(len(CANDIDATES), dtype=np.float64)
    for subject in DEVELOPMENT:
        cv, forward, source_hash = selection_scores(feature_cache, subject)
        total_cv += cv / len(DEVELOPMENT)
        total_forward += forward / len(DEVELOPMENT)
        result["development_sources"][subject] = source_hash
        print("development", subject, flush=True)
    global_cv = tuple(map(int, CANDIDATES[np.argmax(total_cv)]))
    global_forward = tuple(map(int, CANDIDATES[np.argmax(total_forward)]))
    result["global_choices"] = {
        "global_train_cv": global_cv,
        "global_forward": global_forward,
        "selected_development_cv": float(total_cv.max()),
        "selected_development_forward": float(total_forward.max()),
    }
    print("global", global_cv, global_forward, flush=True)
    for subject in METHOD_CHECK:
        cv, forward, source_hash = selection_scores(feature_cache, subject)
        choices = {
            "personal_train_cv": tuple(map(int, CANDIDATES[np.argmax(cv)])),
            "personal_forward": tuple(map(int, CANDIDATES[np.argmax(forward)])),
            "global_train_cv": global_cv,
            "global_forward": global_forward,
            "fixed_index_spread": FIXED,
        }
        result["method_choices"][subject] = {
            "source_member_sha256": source_hash,
            "programs": choices,
            "selected_train_cv": float(cv.max()),
            "selected_forward": float(forward.max()),
        }
        print("method", subject, choices["personal_train_cv"], flush=True)
    write_exclusive(output, result)
    print("selection_sha256", canonical_hash(result), flush=True)


def score(feature_cache: Path, selection_path: Path, output: Path):
    if output.exists():
        raise ValueError("score already exists")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if (selection.get("schema") != "openalterego.faceemg11-method-selection.v1" or
            selection.get("code_sha256") != code_hash() or
            tuple(selection.get("development_subjects", ())) != DEVELOPMENT or
            tuple(selection.get("method_check_subjects", ())) != METHOD_CHECK):
        raise ValueError("selection or code changed")
    result = {"schema": "openalterego.faceemg11-method-score.v1",
              "selection_sha256": canonical_hash(selection), "subjects": {}}
    for subject in METHOD_CHECK:
        features, labels, groups, metadata = load_subject(feature_cache, subject)
        chosen = selection["method_choices"][subject]
        if metadata["member_sha256"] != chosen["source_member_sha256"]:
            raise ValueError(f"source member changed: {subject}")
        train, validation, query = split_subject(features, labels, groups)
        fit = (np.vstack((train[0], validation[0])),
               np.r_[train[1], validation[1]])
        fold = prepare_fold(fit, query, features_per_site=2)
        designs = np.asarray([chosen["programs"][name]
                              for name in CHOICE_NAMES], dtype=np.int16)
        values = score_programs([fold], designs, features_per_site=2)
        all_twenty = float(score_programs(
            [fold], np.arange(20, dtype=np.int16)[None, :],
            features_per_site=2)[0])
        scores = {name: float(value)
                  for name, value in zip(CHOICE_NAMES, values)}
        scores["all_twenty_channel_reference"] = all_twenty
        result["subjects"][subject] = {
            "query_blocks": list(range(21, 31)),
            "query_trials": len(query[1]),
            "query_balanced_accuracy": scores,
            "primary_paired_gain": (scores["personal_train_cv"] -
                                    scores["global_forward"]),
        }
        print(subject, {name: round(value, 3) for name, value in scores.items()},
              flush=True)
    write_exclusive(output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    choose = sub.add_parser("select")
    choose.add_argument("--feature-cache", type=Path, required=True)
    choose.add_argument("--output", type=Path, required=True)
    evaluate = sub.add_parser("score")
    evaluate.add_argument("--feature-cache", type=Path, required=True)
    evaluate.add_argument("--selection", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "select":
        select(args.feature_cache, args.output)
    else:
        score(args.feature_cache, args.selection, args.output)


if __name__ == "__main__":
    main()
