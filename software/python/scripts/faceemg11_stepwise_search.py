"""Beam search six facial channels using a temporal-feature objective."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from openalterego.research.compositional_search import beam_search
from openalterego.research.dense_programs import (
    prepare_cv_folds, prepare_fold, score_programs)
from scripts.faceemg11_multifidelity_search import temporal_subject
from scripts.faceemg11_program_search import (
    load_subject, split_subject)


OPTIONS = tuple(tuple((channel,) for channel in range(20)) for _ in range(6))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse-cache", type=Path, required=True)
    parser.add_argument("--temporal-cache", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--beam-sizes", type=int, nargs="+", default=(1, 4, 16))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    result = {
        "schema": "openalterego.faceemg11-stepwise-search.v1",
        "scope": "Exposed facial-command later-block replay; partial-program CV heuristic",
        "feature_count_per_channel": 8,
        "full_program_space": 38760,
        "beam_sizes": args.beam_sizes,
        "subjects": {},
    }
    for subject in args.subjects:
        _, _, _, metadata = load_subject(args.coarse_cache, subject)
        features, labels, groups = temporal_subject(
            args.temporal_cache, subject, metadata["member_sha256"])
        train, validation, query = split_subject(features, labels, groups)
        folds = prepare_cv_folds(*train, features_per_site=8)
        scorer = lambda designs: score_programs(
            folds, designs, features_per_site=8)
        programs, timings = {}, {}
        for width in args.beam_sizes:
            started = time.perf_counter()
            chosen = beam_search(OPTIONS, scorer, beam_size=width)
            elapsed = time.perf_counter() - started
            name = f"beam_{width}"
            programs[name] = chosen.channels
            timings[name] = {
                "train_cv": chosen.score,
                "evaluations": chosen.evaluations,
                "stage_evaluations": chosen.stage_evaluations,
                "search_seconds": elapsed,
            }
        fit = (np.vstack((train[0], validation[0])),
               np.r_[train[1], validation[1]])
        fold = prepare_fold(fit, query, features_per_site=8)
        values = score_programs(
            [fold], np.asarray(list(programs.values()), dtype=np.int16),
            features_per_site=8)
        result["subjects"][subject] = {
            "source_member_sha256": metadata["member_sha256"],
            "programs": programs, "search": timings,
            "query_balanced_accuracy": {
                name: float(value) for name, value in zip(programs, values)},
        }
        print(subject, {name: {
            "query": round(float(score), 3),
            "evaluations": timings[name]["evaluations"],
            "seconds": round(timings[name]["search_seconds"], 3),
        } for name, score in zip(programs, values)}, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
