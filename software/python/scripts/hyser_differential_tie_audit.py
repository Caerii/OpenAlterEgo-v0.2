"""Audit day-one score plateaus and their next-day outcome spread."""

import argparse
import json
from pathlib import Path

import numpy as np

from openalterego.research.dense_programs import (
    prepare_cv_folds, prepare_fold, score_programs)
from scripts.hyser_differential_program_pilot import candidate_programs, load
from scripts.hyser_pair_stability_pilot import SUBJECTS


def tied_maximum(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("invalid score vector")
    return np.flatnonzero(np.isclose(scores, scores.max(), rtol=0, atol=1e-12))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    result = {"schema": "openalterego.hyser-differential-tie-audit.v1",
              "scope": "Retrospective measured-pair forearm development; future-day labels used only to characterize tied choices",
              "candidate_count": 5000, "candidate_seed": 42,
              "cv_seed": 42, "cv_folds": 3, "alpha": 10.0,
              "tie_tolerance": "absolute 1e-12", "subjects": {}}
    for subject in SUBJECTS:
        train, meta1 = load(args.cache_dir, subject, 1)
        target, meta2 = load(args.cache_dir, subject, 2)
        designs = candidate_programs(meta1, 5000)
        cv = score_programs(prepare_cv_folds(*train, features_per_site=2),
                            designs, features_per_site=2)
        actual = score_programs([prepare_fold(train, target, features_per_site=2)],
                                designs, features_per_site=2)
        ties = tied_maximum(cv)
        result["subjects"][subject] = {
            "day1_source_record_sha256": meta1["source_record_sha256"],
            "day2_source_record_sha256": meta2["source_record_sha256"],
            "maximum_day1_cv": float(cv.max()),
            "tied_maximum_count": len(ties),
            "canonical_first_tie_day2": float(actual[ties[0]]),
            "tied_day2_min": float(actual[ties].min()),
            "tied_day2_median": float(np.median(actual[ties])),
            "tied_day2_mean": float(actual[ties].mean()),
            "tied_day2_max": float(actual[ties].max()),
            "candidate_day2_median": float(np.median(actual)),
            "candidate_day2_oracle": float(actual.max())}
        print(subject, "ties", len(ties), "day2_range",
              round(float(actual[ties].min()), 3),
              round(float(actual[ties].max()), 3), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
