"""Freeze dense-array search choices before acquiring a new-day cohort."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from openalterego.research import hyser_search
from openalterego.research import hyser_features
from openalterego.research.hyser_features import site_coordinates
from openalterego.research.hyser_search import (
    greedy_search, prepare_folds, random_budget_search, ridge_test,
    score_designs)
from scripts import hyser_program_search
from scripts.hyser_program_search import fixed_grid_design, load_session


SCHEMA = "openalterego.hyser-holdout-selection.v1"


def code_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes() +
                           Path(hyser_search.__file__).read_bytes() +
                           Path(hyser_features.__file__).read_bytes() +
                           Path(hyser_program_search.__file__).read_bytes()).hexdigest()


def canonical_sha256(record: dict) -> str:
    return hashlib.sha256(json.dumps(record, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def select(cache: Path, subjects: list[str], output: Path, seed: int):
    if output.exists():
        raise ValueError("selection output exists")
    if any(list(cache.glob(f"subject{subject}-session2-*.npz")) for subject in subjects):
        raise ValueError("future-session feature cache already exists for this cohort")
    result = {"schema": SCHEMA, "code_sha256": code_sha256(), "seed": seed,
              "subjects": subjects,
              "protocol": {"train_session": 1, "evaluation_session": 2,
                           "width": 8, "cv_folds": 3, "ridge_alpha": 10.0,
                           "min_grid_distance_cells": 2,
                           "primary_comparison": "coverage_greedy versus fixed_grid next-day balanced accuracy",
                           "secondary_comparisons": ["unconstrained_greedy", "coverage_random_budget",
                                                     "unconstrained_random_budget", "all_256_reference"]},
              "scope": "Forearm gesture method check; choices frozen in Git before new-day acquisition. Not an independent preregistration.",
              "choices": {}}
    for subject in subjects:
        train, meta = load_session(cache, subject, 1)
        sites = site_coordinates(tuple(meta["site_names"]))
        folds = prepare_folds(*train, seed=seed)
        rows = {}
        for name, cap in (("unconstrained", None), ("coverage", 2)):
            start = time.perf_counter()
            greedy, score, budget = greedy_search(folds, sites, max_per_grid=cap)
            greedy_time = time.perf_counter() - start
            start = time.perf_counter()
            random, random_score = random_budget_search(
                folds, sites, width=8, budget=budget, seed=seed,
                max_per_grid=cap)
            random_time = time.perf_counter() - start
            rows[name + "_greedy"] = {"channels": greedy, "selection_cv": score,
                                       "evaluations": budget, "selection_seconds": greedy_time}
            rows[name + "_random_budget"] = {"channels": random,
                                              "selection_cv": random_score,
                                              "evaluations": budget,
                                              "selection_seconds": random_time}
        fixed = fixed_grid_design(sites)
        rows["fixed_grid"] = {"channels": fixed,
                              "selection_cv": float(score_designs(folds, np.asarray([fixed]))[0]),
                              "evaluations": 1}
        result["choices"][subject] = {"train_source_record_sha256": meta["source_record_sha256"],
                                      "publisher_manifest_sha256": meta["publisher_manifest_sha256"],
                                      "site_names": meta["site_names"],
                                      "train_trials": len(train[1]), "strategies": rows}
        print(subject, {name: round(row["selection_cv"], 3)
                        for name, row in rows.items()}, flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print("selection_sha256", canonical_sha256(result), flush=True)


def score(cache: Path, selection_path: Path, output: Path):
    if output.exists():
        raise ValueError("score output exists")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("schema") != SCHEMA or selection.get("code_sha256") != code_sha256():
        raise ValueError("selection protocol or code differs")
    result = {"schema": "openalterego.hyser-holdout-score.v1",
              "selection_sha256": canonical_sha256(selection),
              "protocol": selection["protocol"], "subjects": {}}
    for subject in selection["subjects"]:
        train, train_meta = load_session(cache, subject, 1)
        prior = selection["choices"][subject]
        if train_meta["source_record_sha256"] != prior["train_source_record_sha256"]:
            raise ValueError(f"train source changed: {subject}")
        test, test_meta = load_session(cache, subject, 2)
        if (test_meta["site_names"] != prior["site_names"] or
                test_meta["publisher_manifest_sha256"] != prior["publisher_manifest_sha256"]):
            raise ValueError(f"next-day geometry or publisher version differs: {subject}")
        scores = {name: ridge_test(train, test, row["channels"])
                  for name, row in prior["strategies"].items()}
        scores["all_256_reference"] = ridge_test(train, test, list(range(256)))
        result["subjects"][subject] = {
            "next_day_balanced_accuracy": scores,
            "train_trials": len(train[1]), "test_trials": len(test[1]),
            "test_source_record_sha256": test_meta["source_record_sha256"]}
        print(subject, scores, flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    select_parser = sub.add_parser("select")
    select_parser.add_argument("--cache-dir", type=Path, required=True)
    select_parser.add_argument("--subjects", nargs="+", required=True)
    select_parser.add_argument("--output", type=Path, required=True)
    select_parser.add_argument("--seed", type=int, default=42)
    score_parser = sub.add_parser("score")
    score_parser.add_argument("--cache-dir", type=Path, required=True)
    score_parser.add_argument("--selection", type=Path, required=True)
    score_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "select":
        if not args.subjects or any(subject not in {f"{index:02d}" for index in range(1, 21)}
                                    for subject in args.subjects):
            parser.error("invalid cohort subjects")
        select(args.cache_dir, args.subjects, args.output, args.seed)
    else:
        score(args.cache_dir, args.selection, args.output)


if __name__ == "__main__":
    main()
