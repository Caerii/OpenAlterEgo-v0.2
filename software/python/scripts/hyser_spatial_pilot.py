"""Development-only test of one-cell translation robust site search."""

import argparse
import json
from pathlib import Path

import numpy as np

from openalterego.research.hyser_features import site_coordinates
from openalterego.research.hyser_search import prepare_folds, ridge_test
from openalterego.research.hyser_spatial_search import (
    candidate_designs, displacement_maps, displacement_scores)
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
    result = {"schema": "openalterego.hyser-spatial-development.v1",
              "scope": "Post-hoc development on subjects with seen next-day outcomes",
              "selection_rule": "best minimum of nominal and four one-grid-cell shifted CV scores",
              "candidate_count": args.candidates, "seed": 42,
              "subjects": {}}
    for subject in args.subjects:
        train, meta = load_session(args.cache_dir, subject, 1)
        test, test_meta = load_session(args.cache_dir, subject, 2)
        sites = site_coordinates(tuple(meta["site_names"]))
        designs = candidate_designs(sites, count=args.candidates)
        scores = displacement_scores(prepare_folds(*train), designs,
                                     displacement_maps(sites))
        picks = {"robust_min": int(np.argmax(scores.min(axis=1))),
                 "nominal_cv": int(np.argmax(scores[:, 0]))}
        strategies = {}
        for name, index in picks.items():
            channels = list(map(int, designs[index]))
            strategies[name] = {"channels": channels,
                                "nominal_cv": float(scores[index, 0]),
                                "minimum_shifted_cv": float(scores[index].min()),
                                "day_two_balanced_accuracy": ridge_test(train, test, channels)}
        fixed = fixed_grid_design(sites)
        strategies["fixed_grid"] = {"channels": fixed,
                                    "day_two_balanced_accuracy": ridge_test(train, test, fixed)}
        result["subjects"][subject] = {
            "train_source_record_sha256": meta["source_record_sha256"],
            "test_source_record_sha256": test_meta["source_record_sha256"],
            "strategies": strategies}
        print(subject, {name: round(row["day_two_balanced_accuracy"], 3)
                        for name, row in strategies.items()}, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
