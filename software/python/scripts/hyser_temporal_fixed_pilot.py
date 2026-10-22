"""Compare whole-trial and time-resolved features on the same fixed eight sites."""

import argparse
import json
from pathlib import Path

import numpy as np

from openalterego.research.dense_programs import (
    prepare_cv_folds, prepare_fold, score_programs)
from openalterego.research.hyser_features import site_coordinates
from scripts.hyser_program_search import fixed_grid_design


ALPHAS = (1.0, 10.0, 100.0, 1000.0)


def load(cache: Path, subject: str, session: int, local_features: int):
    paths = list(cache.glob(f"subject{subject}-session{session}-*.npz"))
    if len(paths) != 1:
        raise ValueError("expected one source-hash-keyed feature cache")
    with np.load(paths[0], allow_pickle=False) as data:
        features = data["features"].astype(np.float64)
        labels = data["labels"].astype(np.int64)
    meta = json.loads(paths[0].with_suffix(".json").read_text())
    if features.shape != (len(labels), 256 * local_features):
        raise ValueError("feature cache shape differs")
    return (features, labels), meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-cache", type=Path, required=True)
    parser.add_argument("--temporal-cache", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    result = {"schema": "openalterego.hyser-temporal-fixed-pilot.v1",
              "scope": "Development subjects with already seen day-two outcomes; no electrode search",
              "fixed_layout": "grid positions (2,2) and (6,6) in each of four grids",
              "alpha_candidates": ALPHAS, "subjects": {}}
    for subject in args.subjects:
        rows = {}
        expected_identity = None
        for name, cache, count in (("whole_trial", args.baseline_cache, 2),
                                   ("four_bins_plus_whole", args.temporal_cache, 10)):
            train, meta = load(cache, subject, 1, count)
            test, test_meta = load(cache, subject, 2, count)
            source_identity = [meta["source_record_sha256"],
                               test_meta["source_record_sha256"]]
            if expected_identity is None:
                expected_identity = source_identity
            elif source_identity != expected_identity:
                raise ValueError("feature families have different source recordings")
            sites = site_coordinates(tuple(meta["site_names"]))
            fixed = np.asarray([fixed_grid_design(sites)], dtype=np.int16)
            folds = prepare_cv_folds(*train, features_per_site=count)
            cv = {str(alpha): float(score_programs(
                folds, fixed, features_per_site=count, alpha=alpha)[0])
                for alpha in ALPHAS}
            best_alpha = min(ALPHAS, key=lambda alpha: (-cv[str(alpha)], alpha))
            day_pair = prepare_fold(train, test, features_per_site=count)
            rows[name] = {"day_one_cv_by_alpha": cv,
                          "chosen_alpha": best_alpha,
                          "day_two_balanced_accuracy": float(score_programs(
                              [day_pair], fixed, features_per_site=count,
                              alpha=best_alpha)[0]),
                          "day_two_at_alpha10": float(score_programs(
                              [day_pair], fixed, features_per_site=count,
                              alpha=10.0)[0])}
        result["subjects"][subject] = {"source_record_sha256": expected_identity,
                                      "features": rows}
        print(subject, {name: round(row["day_two_balanced_accuracy"], 3)
                        for name, row in rows.items()}, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
