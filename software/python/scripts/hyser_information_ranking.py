"""Post-hoc test of cheap day-one information scores against next-day decoding."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from openalterego.research.hyser_features import site_coordinates
from openalterego.research.hyser_information_proxy import class_covariances
from openalterego.research.hyser_information_proxy import score_designs as information_scores
from openalterego.research.hyser_search import prepare_folds, score_designs as cv_scores
from openalterego.research.hyser_spatial_search import candidate_designs
from openalterego.research.hyser_transfer import prepare_day_pair, score_day_pair
from scripts.hyser_program_search import fixed_grid_design, load_session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates", type=int, default=1000)
    parser.add_argument("--ridge", type=float, default=0.25)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    result = {"schema": "openalterego.hyser-information-ranking.v1",
              "scope": "Post-hoc forearm development audit; day-two labels are used only for evaluation",
              "candidate_count": args.candidates, "candidate_seed": 42,
              "ridge": args.ridge, "subjects": {}}
    for subject in args.subjects:
        day1, meta1 = load_session(args.cache_dir, subject, 1)
        day2, meta2 = load_session(args.cache_dir, subject, 2)
        if tuple(meta1["site_names"]) != tuple(meta2["site_names"]):
            raise ValueError(f"channel labels differ across sessions: {subject}")
        sites = site_coordinates(tuple(meta1["site_names"]))
        designs = candidate_designs(sites, count=args.candidates)
        between, within = class_covariances(*day1)
        info = information_scores(between, within, designs, ridge=args.ridge)
        cv = cv_scores(prepare_folds(*day1), designs)
        actual = score_day_pair(prepare_day_pair(day1, day2), designs)
        fixed = np.asarray([fixed_grid_design(sites)], dtype=np.int16)
        fixed_actual = float(score_day_pair(prepare_day_pair(day1, day2), fixed)[0])
        predictors = {}
        for name, proxy in (("information", info), ("day1_cv", cv)):
            chosen = int(np.argmax(proxy))
            shortlist_size = max(1, int(np.ceil(0.1 * len(designs))))
            predicted_top = np.argsort(proxy, kind="stable")[-shortlist_size:]
            actual_top = np.argsort(actual, kind="stable")[-shortlist_size:]
            predictors[name] = {
                "spearman_with_day2": float(spearmanr(proxy, actual).statistic),
                "top10_percent_recall": float(len(set(predicted_top) &
                                                  set(actual_top)) / shortlist_size),
                "best_day2_in_top10_percent": float(actual[predicted_top].max()),
                "selected_channels": designs[chosen].astype(int).tolist(),
                "selected_proxy": float(proxy[chosen]),
                "selected_day2_balanced_accuracy": float(actual[chosen]),
                "selection_regret_vs_pool_oracle": float(actual.max() - actual[chosen]),
                "selected_minus_fixed": float(actual[chosen] - fixed_actual)}
        result["subjects"][subject] = {
            "day1_source_record_sha256": meta1["source_record_sha256"],
            "day2_source_record_sha256": meta2["source_record_sha256"],
            "fixed_day2_balanced_accuracy": fixed_actual,
            "candidate_median_day2": float(np.median(actual)),
            "candidate_oracle_day2": float(actual.max()),
            "predictors": predictors}
        print(subject, {name: (round(row["spearman_with_day2"], 3),
                               round(row["selected_minus_fixed"], 3))
                        for name, row in predictors.items()}, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
