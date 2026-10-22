"""Measure how differential program-search budget affects next-day decisions."""

import argparse
import json
from pathlib import Path

import numpy as np

from openalterego.research.dense_programs import (
    prepare_cv_folds, prepare_fold, score_programs)
from scripts.hyser_differential_program_pilot import (
    candidate_programs, fixed_program, load)


def budget_curve(day1_scores: np.ndarray, day2_scores: np.ndarray,
                 budgets: tuple[int, ...], repetitions: int, seed: int) -> dict:
    """Use the same random order for nested budgets within each repetition."""
    day1_scores = np.asarray(day1_scores, dtype=np.float64)
    day2_scores = np.asarray(day2_scores, dtype=np.float64)
    if (day1_scores.ndim != 1 or day1_scores.shape != day2_scores.shape or
            not len(day1_scores) or not np.isfinite(day1_scores).all() or
            not np.isfinite(day2_scores).all() or repetitions < 1 or
            any(b < 1 or b > len(day1_scores) for b in budgets)):
        raise ValueError("invalid score array or search budgets")
    rng = np.random.default_rng(seed)
    selected_cv = np.empty((repetitions, len(budgets)))
    selected_day2 = np.empty_like(selected_cv)
    random_day2 = np.empty(repetitions)
    for repetition in range(repetitions):
        ordering = rng.permutation(len(day1_scores))
        random_day2[repetition] = day2_scores[ordering[0]]
        for column, budget in enumerate(budgets):
            candidates = ordering[:budget]
            selected = candidates[np.argmax(day1_scores[candidates])]
            selected_cv[repetition, column] = day1_scores[selected]
            selected_day2[repetition, column] = day2_scores[selected]
    return {"budgets": list(budgets),
            "repetitions": repetitions,
            "mean_day1_cv": selected_cv.mean(axis=0).tolist(),
            "mean_selected_day2": selected_day2.mean(axis=0).tolist(),
            "sd_selected_day2": selected_day2.std(axis=0).tolist(),
            "mean_random_one_day2": float(random_day2.mean()),
            "mean_selected_minus_random_one_day2": (
                selected_day2 - random_day2[:, None]).mean(axis=0).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--candidates", type=int, default=5000)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    budgets = (1, 10, 30, 100, 300, 1000, 3000, args.candidates)
    if args.candidates < 3000:
        parser.error("candidates must be at least 3000")
    result = {"schema": "openalterego.hyser-differential-budget-audit.v1",
              "scope": "Retrospective development on measured digital forearm pairs; repeated candidate orderings are not independent people",
              "candidate_seed": 42, "ordering_seed": 991,
              "candidate_count": args.candidates, "alpha": 10.0,
              "subjects": {}}
    for subject in args.subjects:
        train, meta1 = load(args.cache_dir, subject, 1)
        target, meta2 = load(args.cache_dir, subject, 2)
        if meta1["pairs"] != meta2["pairs"]:
            raise ValueError("pair map changed between days")
        designs = candidate_programs(meta1, args.candidates)
        cv = score_programs(prepare_cv_folds(*train, features_per_site=2),
                            designs, features_per_site=2)
        fold = prepare_fold(train, target, features_per_site=2)
        actual = score_programs([fold], designs, features_per_site=2)
        fixed = fixed_program(meta1)
        fixed_score = float(score_programs([fold], np.asarray([fixed]),
                                           features_per_site=2)[0])
        result["subjects"][subject] = {
            "day1_source_record_sha256": meta1["source_record_sha256"],
            "day2_source_record_sha256": meta2["source_record_sha256"],
            "fixed_day2": fixed_score,
            "candidate_day2_median": float(np.median(actual)),
            "candidate_day2_oracle": float(actual.max()),
            "curve": budget_curve(cv, actual, budgets, args.repetitions, 991)}
        print(subject, "fixed", round(fixed_score, 3),
              "budget_last", round(result["subjects"][subject]["curve"][
                  "mean_selected_day2"][-1], 3), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
