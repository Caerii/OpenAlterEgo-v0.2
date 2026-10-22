"""Compare two-stage facial search with exact temporal CV search."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from openalterego.research.dense_programs import (
    prepare_cv_folds, prepare_fold, score_programs)
from scripts.faceemg11_multifidelity_search import temporal_subject
from scripts.faceemg11_program_search import (
    CANDIDATES, load_subject, split_subject)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse-cache", type=Path, required=True)
    parser.add_argument("--temporal-cache", type=Path, required=True)
    parser.add_argument("--two-stage-result", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    two_stage = json.loads(args.two_stage_result.read_text(encoding="utf-8"))
    if two_stage.get("schema") != "openalterego.faceemg11-multifidelity-search.v1":
        raise ValueError("unexpected two-stage artifact")
    result = {"schema": "openalterego.faceemg11-exact-temporal-benchmark.v1",
              "scope": "Development-only timing and selection-regret check; no new held-out claim",
              "candidate_count": len(CANDIDATES), "subjects": {}}
    for subject in args.subjects:
        _, _, _, coarse_meta = load_subject(args.coarse_cache, subject)
        features, labels, groups = temporal_subject(
            args.temporal_cache, subject, coarse_meta["member_sha256"])
        train, validation, query = split_subject(features, labels, groups)
        started = time.perf_counter()
        cv = score_programs(prepare_cv_folds(*train, features_per_site=8),
                            CANDIDATES, features_per_site=8)
        exact_seconds = time.perf_counter() - started
        exact_index = int(np.argmax(cv))
        exact_program = tuple(map(int, CANDIDATES[exact_index]))
        finalist_program = tuple(two_stage["subjects"][subject]["programs"][
            "temporal_reranked"])
        matches = np.flatnonzero(np.all(
            CANDIDATES == np.asarray(finalist_program, dtype=np.int16), axis=1))
        if len(matches) != 1:
            raise ValueError("two-stage finalist absent from candidate grammar")
        finalist_index = int(matches[0])
        fit = (np.vstack((train[0], validation[0])),
               np.r_[train[1], validation[1]])
        fold = prepare_fold(fit, query, features_per_site=8)
        query_scores = score_programs(
            [fold], np.asarray([exact_program, finalist_program], dtype=np.int16),
            features_per_site=8)
        prior = two_stage["subjects"][subject]
        result["subjects"][subject] = {
            "source_member_sha256": coarse_meta["member_sha256"],
            "exact_program": exact_program,
            "two_stage_program": finalist_program,
            "exact_cv": float(cv[exact_index]),
            "two_stage_cv": float(cv[finalist_index]),
            "cv_regret": float(cv[exact_index] - cv[finalist_index]),
            "exact_query_balanced_accuracy": float(query_scores[0]),
            "two_stage_query_balanced_accuracy": float(query_scores[1]),
            "exact_cv_seconds": exact_seconds,
            "two_stage_search_seconds": (prior["low_screen_seconds"] +
                                          prior["high_rerank_seconds"]),
        }
        print(subject, "exact", round(float(cv[exact_index]), 3),
              "two-stage", round(float(cv[finalist_index]), 3),
              "exact_seconds", round(exact_seconds, 2), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
