"""LOO development audit of a compositional donor drift prior for pair search."""

import argparse
import json
from pathlib import Path

import numpy as np

from openalterego.research.dense_programs import (
    prepare_cv_folds, prepare_fold, score_programs)
from openalterego.research.hyser_pair_stability import (
    pair_drift_ranks, rank_programs)
from scripts.hyser_differential_program_pilot import (
    candidate_programs, fixed_program, load)


SUBJECTS = ("01", "02", "03", "04", "05", "06", "07", "08",
            "17", "18", "19", "20")
PENALTIES = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    sessions = {}
    for subject in SUBJECTS:
        train, meta1 = load(args.cache_dir, subject, 1)
        target, meta2 = load(args.cache_dir, subject, 2)
        if meta1["pairs"] != meta2["pairs"] or meta1["site_names"] != meta2["site_names"]:
            raise ValueError(f"pair metadata changed: {subject}")
        sessions[subject] = (train, target, meta1, meta2)
    reference = sessions[SUBJECTS[0]][2]
    if any(sessions[subject][2]["pairs"] != reference["pairs"] or
           sessions[subject][2]["site_names"] != reference["site_names"]
           for subject in SUBJECTS):
        raise ValueError("donor pair maps differ")
    individual = {subject: pair_drift_ranks(
        sessions[subject][0], sessions[subject][1], reference["pairs"],
        reference["site_names"]) for subject in SUBJECTS}
    result = {"schema": "openalterego.hyser-pair-stability-loo.v1",
              "scope": "Retrospective 12-person development; donor day-two features exclude target person, target day-two labels score only",
              "subjects": {}, "penalties": PENALTIES,
              "candidate_count": 5000, "candidate_seed": 42,
              "pair_prior": "class-centroid day drift divided by day-one class signal plus 0.5, within-grid percentile, mean of other 11 people"}
    for subject in SUBJECTS:
        train, target, meta1, meta2 = sessions[subject]
        donors = [name for name in SUBJECTS if name != subject]
        prior = np.mean([individual[name] for name in donors], axis=0)
        designs = candidate_programs(meta1, 5000)
        cv = score_programs(prepare_cv_folds(*train, features_per_site=2),
                            designs, features_per_site=2)
        fold = prepare_fold(train, target, features_per_site=2)
        actual = score_programs([fold], designs, features_per_site=2)
        fixed = float(score_programs([fold], np.asarray([fixed_program(meta1)]),
                                     features_per_site=2)[0])
        rows = {}
        for penalty in PENALTIES:
            ranking = rank_programs(cv, designs, prior, penalty)
            chosen = int(np.argmax(ranking))
            rows[str(penalty)] = {"channels": designs[chosen].astype(int).tolist(),
                                  "day1_cv": float(cv[chosen]),
                                  "donor_drift_rank": float(prior[designs[chosen]].mean()),
                                  "day2_balanced_accuracy": float(actual[chosen]),
                                  "minus_fixed_differential": float(actual[chosen] - fixed)}
        result["subjects"][subject] = {
            "day1_source_record_sha256": meta1["source_record_sha256"],
            "day2_source_record_sha256": meta2["source_record_sha256"],
            "donors": donors,
            "fixed_differential_day2": fixed,
            "candidate_day2_oracle": float(actual.max()),
            "choices": rows}
        print(subject, {str(penalty): round(rows[str(penalty)][
            "day2_balanced_accuracy"], 3) for penalty in PENALTIES}, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
