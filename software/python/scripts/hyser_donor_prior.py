"""Leave-one-person-out audit of a shared next-day layout prior."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from openalterego.research.hyser_features import site_coordinates
from openalterego.research.hyser_search import ridge_test
from openalterego.research.hyser_spatial_search import candidate_designs
from openalterego.research.hyser_transfer import prepare_day_pair, score_day_pair
from scripts.hyser_program_search import fixed_grid_design, load_session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates", type=int, default=1000)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    subjects = args.subjects
    outcomes, fixed_scores, source_hashes = [], [], {}
    designs = None
    expected_sites = None
    for subject in subjects:
        train, train_meta = load_session(args.cache_dir, subject, 1)
        test, test_meta = load_session(args.cache_dir, subject, 2)
        if expected_sites is None:
            expected_sites = train_meta["site_names"]
            sites = site_coordinates(tuple(expected_sites))
            designs = candidate_designs(sites, count=args.candidates)
        elif train_meta["site_names"] != expected_sites:
            raise ValueError("site order differs between subjects")
        outcomes.append(score_day_pair(prepare_day_pair(train, test), designs))
        fixed_scores.append(ridge_test(train, test, fixed_grid_design(sites)))
        source_hashes[subject] = [train_meta["source_record_sha256"],
                                  test_meta["source_record_sha256"]]
    outcomes = np.asarray(outcomes)
    rows = {}
    for index, subject in enumerate(subjects):
        donor_mean = np.delete(outcomes, index, axis=0).mean(axis=0)
        chosen = int(np.argmax(donor_mean))
        rows[subject] = {
            "donor_mean_spearman_with_next_day": float(spearmanr(donor_mean, outcomes[index]).statistic),
            "donor_chosen_channels": list(map(int, designs[chosen])),
            "donor_chosen_next_day": float(outcomes[index, chosen]),
            "fixed_grid_next_day": float(fixed_scores[index]),
            "candidate_median_next_day": float(np.median(outcomes[index])),
            "candidate_oracle_best_next_day": float(outcomes[index].max())}
        print(subject, round(rows[subject]["donor_mean_spearman_with_next_day"], 3),
              round(rows[subject]["donor_chosen_next_day"], 3),
              round(rows[subject]["fixed_grid_next_day"], 3), flush=True)
    result = {"schema": "openalterego.hyser-donor-prior-development.v1",
              "scope": "Post-hoc leave-one-person-out audit on subjects with seen outcomes",
              "subjects": subjects, "candidate_count": args.candidates, "seed": 42,
              "source_record_sha256": source_hashes, "results": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
