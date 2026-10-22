"""Development replay with labeled day-two support and disjoint query trials."""

import argparse
import json
from pathlib import Path

import numpy as np

from openalterego.research.hyser_features import site_coordinates
from openalterego.research.hyser_fewshot import ridge_query_score, support_query
from scripts.hyser_program_search import fixed_grid_design
from scripts.hyser_temporal_fixed_pilot import load


WEIGHTS = (0.0, 0.25, 1.0, 4.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-cache", type=Path, required=True)
    parser.add_argument("--temporal-cache", type=Path, required=True)
    parser.add_argument("--alpha-record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    alpha_record = json.loads(args.alpha_record.read_text(encoding="utf-8"))
    result = {"schema": "openalterego.hyser-fewshot-development.v1",
              "scope": "Post-hoc support/query by within-class repetition; class-block recording order is not chronological calibration before all queries",
              "support_policy": "first one or two recorded repetitions per class; all remaining same-session repetitions form query",
              "support_weights": WEIGHTS,
              "alpha_source": str(args.alpha_record),
              "subjects": {}}
    for subject, earlier in alpha_record["subjects"].items():
        rows = {}
        for name, cache, count in (("whole_trial", args.baseline_cache, 2),
                                   ("four_bins_plus_whole", args.temporal_cache, 10)):
            train, meta = load(cache, subject, 1, count)
            test, test_meta = load(cache, subject, 2, count)
            if [meta["source_record_sha256"], test_meta["source_record_sha256"]] != earlier["source_record_sha256"]:
                raise ValueError("source hashes differ from alpha selection")
            sites = site_coordinates(tuple(meta["site_names"]))
            fixed = fixed_grid_design(sites)
            alpha = earlier["features"][name]["chosen_alpha"]
            rows[name] = {}
            for shots in (1, 2):
                support, query = support_query(test[1], shots)
                rows[name][str(shots)] = {
                    "support_trials": len(support), "query_trials": len(query),
                    "query_balanced_accuracy_by_support_weight": {
                        str(weight): ridge_query_score(
                            train, test, fixed, features_per_site=count,
                            alpha=alpha, support=support, query=query,
                            support_weight=weight)
                        for weight in WEIGHTS}}
        result["subjects"][subject] = rows
        print(subject, {name: round(rows[name]["1"]["query_balanced_accuracy_by_support_weight"]["1.0"], 3)
                        for name in rows}, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
