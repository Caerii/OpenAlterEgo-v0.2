"""Development-only next-day check of a day-one grid-neighbor contact veto."""

import argparse
import json
from pathlib import Path

from openalterego.research.hyser_features import site_coordinates
from openalterego.research.hyser_search import prepare_folds, ridge_test
from openalterego.research.hyser_site_health import dark_contact_mask, screened_greedy
from scripts.hyser_program_search import fixed_grid_design, load_session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log-rms-gap", type=float, default=0.5)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    result = {"schema": "openalterego.hyser-site-health-development.v1",
              "scope": "Post-hoc development; subjects 01-05 day-two outcomes have already been seen",
              "log_rms_gap": args.log_rms_gap, "subjects": {}}
    for subject in args.subjects:
        train, meta = load_session(args.cache_dir, subject, 1)
        test, test_meta = load_session(args.cache_dir, subject, 2)
        sites = site_coordinates(tuple(meta["site_names"]))
        allowed, darkness = dark_contact_mask(train[0], sites, log_rms_gap=args.log_rms_gap)
        folds = prepare_folds(*train)
        strategies = {}
        for name, cap in (("screened_greedy", None), ("screened_coverage", 2)):
            channels, cv, count = screened_greedy(folds, sites, allowed, max_per_grid=cap)
            strategies[name] = {"channels": channels, "day_one_cv": cv,
                                "day_two_balanced_accuracy": ridge_test(train, test, channels),
                                "candidate_evaluations": count,
                                "max_day_one_darkness": float(max(darkness[channels]))}
        fixed = fixed_grid_design(sites)
        strategies["fixed_grid"] = {"channels": fixed,
                                    "day_two_balanced_accuracy": ridge_test(train, test, fixed)}
        result["subjects"][subject] = {
            "train_source_record_sha256": meta["source_record_sha256"],
            "test_source_record_sha256": test_meta["source_record_sha256"],
            "vetoed_sites": [int(index) for index in range(256) if not allowed[index]],
            "strategies": strategies}
        print(subject, {name: round(row["day_two_balanced_accuracy"], 3)
                        for name, row in strategies.items()}, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
