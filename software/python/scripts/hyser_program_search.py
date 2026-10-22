"""Search 256 recorded Hyser sites for eight-channel next-day gesture decoding."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from openalterego.research import hyser_search
from openalterego.research.hyser_features import site_coordinates
from openalterego.research.hyser_search import (
    feasible, greedy_search, prepare_folds, random_budget_search, ridge_test,
    score_designs)


def load_session(cache: Path, subject: str, session: int):
    paths = list(cache.glob(f"subject{subject}-session{session}-*.npz"))
    if len(paths) != 1:
        raise ValueError(f"expected exactly one verified feature cache: {subject}/{session}")
    with np.load(paths[0], allow_pickle=False) as source:
        features = source["features"].astype(np.float64)
        labels = source["labels"].astype(np.int64)
    metadata = json.loads(paths[0].with_suffix(".json").read_text())
    if (features.shape != (metadata["trials"], 512) or
            labels.shape != (len(features),) or
            not np.isfinite(features).all()):
        raise ValueError(f"invalid feature cache: {paths[0]}")
    return (features, labels), metadata


def fixed_grid_design(sites: np.ndarray) -> list[int]:
    lookup = {tuple(map(int, site)): index for index, site in enumerate(sites)}
    return [lookup[(grid, row, column)] for grid in range(4)
            for row, column in ((1, 1), (5, 5))]


def freeze(path: Path, record: dict) -> str:
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(canonical).hexdigest()
    if path.exists():
        if json.loads(path.read_text()) != record:
            raise ValueError(f"selection record differs: {path}")
        return digest
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        json.dump(record, output, indent=2, allow_nan=False)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    return digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-quota", action="store_true",
                        help="select exactly two sites per grid for eight-site designs")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if any(subject not in {f"{index:02d}" for index in range(1, 21)} for subject in args.subjects):
        parser.error("invalid Hyser subjects")
    config = {"subjects": args.subjects, "seed": args.seed, "width": 8,
              "min_grid_distance_cells": 2, "cv_folds": 3,
              "ridge_alpha": 10.0, "features": "whole-trial log RMS and waveform length",
              "max_per_grid": 2 if args.grid_quota else None}
    if args.output.exists():
        if not args.resume:
            parser.error("output exists; use --resume")
        result = json.loads(args.output.read_text())
        if result.get("schema") != "openalterego.hyser-program-search.v2" or result.get("config") != config:
            parser.error("existing output configuration differs")
    else:
        result = {"schema": "openalterego.hyser-program-search.v2", "config": config,
                  "scope": "256 recorded forearm electrodes; next-day gesture transfer tests the search method, not facial silent speech or unrecorded positions.",
                  "subjects": {}}
    code_hash = hashlib.sha256(Path(__file__).read_bytes() +
                               Path(hyser_search.__file__).read_bytes()).hexdigest()
    for subject in args.subjects:
        if subject in result["subjects"]:
            continue
        train, meta = load_session(args.cache_dir, subject, 1)
        sites = site_coordinates(tuple(meta["site_names"]))
        folds = prepare_folds(*train, seed=args.seed)
        started = time.perf_counter()
        grid_cap = 2 if args.grid_quota else None
        greedy, greedy_cv, evaluated = greedy_search(folds, sites,
                                                     max_per_grid=grid_cap)
        greedy_seconds = time.perf_counter() - started
        started = time.perf_counter()
        random_best, random_cv = random_budget_search(
            folds, sites, width=8, budget=evaluated, seed=args.seed,
            max_per_grid=grid_cap)
        random_seconds = time.perf_counter() - started
        fixed = fixed_grid_design(sites)
        rng = np.random.default_rng(args.seed)
        while True:
            random_one = list(map(int, rng.choice(256, size=8, replace=False)))
            if feasible(tuple(random_one), sites, max_per_grid=grid_cap):
                break
        choices = {"greedy": {"channels": greedy, "cv_balanced_accuracy": greedy_cv},
                   "random_budget": {"channels": random_best,
                                     "cv_balanced_accuracy": random_cv},
                   "fixed_grid": {"channels": fixed,
                                  "cv_balanced_accuracy": float(score_designs(folds, np.asarray([fixed]))[0])},
                   "random_one": {"channels": random_one,
                                  "cv_balanced_accuracy": float(score_designs(folds, np.asarray([random_one]))[0])}}
        record = {"schema": "openalterego.hyser-program-selection.v2",
                  "subject": subject, "code_sha256": code_hash,
                  "input_source_record_sha256": meta["source_record_sha256"],
                  "seed": args.seed, "objective_evaluations_each": evaluated,
                  "choices": choices,
                  "scope": "Local exclusive record before loading session 2; not an independent external timestamp."}
        selection_path = args.output.with_name(f"{args.output.stem}-{subject}-selection.json")
        selection_sha256 = freeze(selection_path, record)
        test, test_meta = load_session(args.cache_dir, subject, 2)
        if tuple(meta["site_names"]) != tuple(test_meta["site_names"]):
            raise ValueError(f"channel labels differ across days: {subject}")
        scores = {name: ridge_test(train, test, row["channels"])
                  for name, row in choices.items()}
        scores["all_256_reference"] = ridge_test(train, test, list(range(256)))
        result["subjects"][subject] = {
            "choices": choices, "next_day_balanced_accuracy": scores,
            "greedy_selection_seconds": greedy_seconds,
            "random_selection_seconds": random_seconds,
            "objective_evaluations_each": evaluated,
            "train_source_record_sha256": meta["source_record_sha256"],
            "test_source_record_sha256": test_meta["source_record_sha256"],
            "train_trials": len(train[1]), "test_trials": len(test[1]),
            "selection_record": str(selection_path),
            "selection_sha256": selection_sha256}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f"{args.output.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                             encoding="utf-8")
        os.replace(temporary, args.output)
        print(subject, scores, f"greedy={greedy_seconds:.2f}s random={random_seconds:.2f}s",
              flush=True)


if __name__ == "__main__":
    main()
