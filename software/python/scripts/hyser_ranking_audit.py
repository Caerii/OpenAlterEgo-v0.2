"""Audit whether day-one design scores rank next-day outcomes across programs."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from openalterego.research.hyser_features import site_coordinates
from openalterego.research.hyser_search import prepare_folds, ridge_test
from openalterego.research.hyser_site_health import dark_contact_mask
from openalterego.research.hyser_spatial_search import (
    candidate_designs, displacement_maps, displacement_scores)
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
    result = {"schema": "openalterego.hyser-ranking-audit.v1",
              "scope": "Post-hoc subjects with seen next-day outcomes; candidate-pool audit, not prospective selection",
              "candidate_count": args.candidates, "seed": 42,
              "subjects": {}}
    for subject in args.subjects:
        train, meta = load_session(args.cache_dir, subject, 1)
        test, test_meta = load_session(args.cache_dir, subject, 2)
        sites = site_coordinates(tuple(meta["site_names"]))
        designs = candidate_designs(sites, count=args.candidates)
        scores = displacement_scores(prepare_folds(*train), designs,
                                     displacement_maps(sites))
        actual = score_day_pair(prepare_day_pair(train, test), designs)
        _, darkness = dark_contact_mask(train[0], sites)
        predictors = {"nominal_cv": scores[:, 0],
                      "mean_shift_cv": scores.mean(axis=1),
                      "minimum_shift_cv": scores.min(axis=1),
                      "darkness": darkness[designs].mean(axis=1)}
        rows = {}
        for name, proxy in predictors.items():
            rank = np.argsort(proxy, kind="stable")
            selected = int(rank[0] if name == "darkness" else rank[-1])
            rows[name] = {"spearman_with_next_day": float(spearmanr(proxy, actual).statistic),
                          "chosen_channels": list(map(int, designs[selected])),
                          "chosen_proxy": float(proxy[selected]),
                          "chosen_next_day": float(actual[selected])}
        fixed = fixed_grid_design(sites)
        result["subjects"][subject] = {
            "train_source_record_sha256": meta["source_record_sha256"],
            "test_source_record_sha256": test_meta["source_record_sha256"],
            "candidate_actual_median": float(np.median(actual)),
            "candidate_actual_best": float(actual.max()),
            "fixed_grid_next_day": ridge_test(train, test, fixed),
            "predictors": rows}
        print(subject, {name: (round(row["spearman_with_next_day"], 3),
                               round(row["chosen_next_day"], 3))
                        for name, row in rows.items()}, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
