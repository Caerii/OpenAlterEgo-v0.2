"""Development replay: select recorded programs with disjoint labeled support."""

import argparse
import json
from pathlib import Path

import numpy as np

from openalterego.research.hyser_features import site_coordinates
from openalterego.research.hyser_fewshot import ridge_query_score, support_query
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
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    result = {"schema": "openalterego.hyser-calibrated-program-search.v1",
              "scope": "Post-hoc forearm development; class-block support/query split is not globally chronological",
              "candidate_count": args.candidates, "seed": 42, "alpha": 10.0,
              "shots_per_class": 1, "support_weight": 1.0,
              "support_gate_margin": 0.2,
              "selection": "Day-one CV, day-one-trained accuracy on labeled day-two support, or their half-weighted average; query labels excluded from selection",
              "subjects": {}}
    for subject in args.subjects:
        day1, meta1 = load_session(args.cache_dir, subject, 1)
        day2, meta2 = load_session(args.cache_dir, subject, 2)
        if tuple(meta1["site_names"]) != tuple(meta2["site_names"]):
            raise ValueError(f"site names changed: {subject}")
        sites = site_coordinates(tuple(meta1["site_names"]))
        designs = candidate_designs(sites, count=args.candidates)
        fixed = fixed_grid_design(sites)
        support, query = support_query(day2[1], 1)
        cv = cv_scores(prepare_folds(*day1), designs)
        support_pair = prepare_day_pair(day1, (day2[0][support], day2[1][support]))
        support_accuracy = score_day_pair(support_pair, designs)
        selection_scores = {"day1_cv": cv,
                            "support_accuracy": support_accuracy,
                            "half_cv_half_support": 0.5 * cv + 0.5 * support_accuracy}
        choices = {"fixed": fixed}
        for name, scores in selection_scores.items():
            choices[name] = designs[int(np.argmax(scores))].astype(int).tolist()
        fixed_support = float(score_day_pair(
            support_pair, np.asarray([fixed], dtype=np.int16))[0])
        best_support = float(np.max(support_accuracy))
        choices["support_gate_0_2"] = (choices["support_accuracy"]
                                       if best_support >= fixed_support + 0.2
                                       else fixed)
        rows = {}
        for name, channels in choices.items():
            rows[name] = {
                "channels": channels,
                "support_accuracy_before_adaptation": float(score_day_pair(
                    support_pair, np.asarray([channels], dtype=np.int16))[0]),
                "query_no_support": ridge_query_score(day1, day2, channels,
                    features_per_site=2, alpha=10.0, support=support,
                    query=query, support_weight=0.0),
                "query_with_support": ridge_query_score(day1, day2, channels,
                    features_per_site=2, alpha=10.0, support=support,
                    query=query, support_weight=1.0)}
        result["subjects"][subject] = {
            "day1_source_record_sha256": meta1["source_record_sha256"],
            "day2_source_record_sha256": meta2["source_record_sha256"],
            "support_trials": len(support), "query_trials": len(query),
            "choices": rows}
        print(subject, {name: round(row["query_with_support"], 3)
                        for name, row in rows.items()}, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
