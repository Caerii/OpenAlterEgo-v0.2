"""Exploratory two-support robust search over measured Hyser programs.

All participants have been used in earlier research. The classwise first two
day-two repetitions provide support, and all remaining repetitions are query.
No query feature or label participates in selection or fitting.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from openalterego.research.dense_programs import (
    prepare_cv_folds, prepare_fold, score_programs)
from openalterego.research.hyser_features import site_coordinates
from openalterego.research.hyser_fewshot import support_query
from openalterego.research.supported_programs import prepare_supported_fold
from scripts.hyser_compositional_search import edge_mode_options
from scripts.hyser_mixed_program_pilot import fixed_mixed, load_mixed, physical_contacts
from scripts.hyser_program_search import fixed_grid_design
from scripts.hyser_support_conditioned_search import sample_programs


BUDGETS = (100, 300, 1000, 3000, 5000)
BUDGET_DRAWS = 10


def two_support_split(labels: np.ndarray):
    support, query = support_query(labels, 2)
    first, second = [], []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        first.append(int(indices[0]))
        second.append(int(indices[1]))
    if ((set(first) | set(second)) != set(map(int, support)) or
            set(first) & set(second) or set(support) & set(query)):
        raise ValueError("support rotations are not disjoint from query")
    return (np.asarray(first), np.asarray(second), support, query)


def nested_budget_indices(scores: np.ndarray, *, seed: int = 991,
                          draws: int = BUDGET_DRAWS) -> dict[int, list[int]]:
    """Select nested seeded prefixes; ties use sorted-program index."""
    if scores.ndim != 1 or len(scores) < max(BUDGETS) or draws < 1:
        raise ValueError("insufficient scored candidates for budget replay")
    rng = np.random.default_rng(seed)
    result = {budget: [] for budget in BUDGETS}
    for _ in range(draws):
        permutation = rng.permutation(len(scores))
        for budget in BUDGETS:
            indices = np.sort(permutation[:budget])
            winner = indices[int(np.argmax(scores[indices]))]
            result[budget].append(int(winner))
    return result


def analyze_subject(mono_cache: Path, pair_cache: Path, subject: str,
                    candidate_count: int) -> dict:
    start = time.perf_counter()
    day1, meta1 = load_mixed(mono_cache, pair_cache, subject, 1)
    day2, meta2 = load_mixed(mono_cache, pair_cache, subject, 2)
    if (meta1["source_record_sha256"] == meta2["source_record_sha256"] or
            meta1["site_names"] != meta2["site_names"] or
            meta1["pairs"] != meta2["pairs"]):
        raise ValueError(f"invalid paired recording alignment: {subject}")
    first_ids, second_ids, support_ids, query_ids = two_support_split(day2[1])
    candidates = sample_programs(edge_mode_options(meta1), candidate_count, 42)
    cv = score_programs(prepare_cv_folds(*day1, features_per_site=2),
                        candidates, features_per_site=2)
    transfer = []
    for ids in (first_ids, second_ids):
        rotated = (day2[0][ids], day2[1][ids])
        transfer.append(score_programs(
            [prepare_fold(day1, rotated, features_per_site=2)],
            candidates, features_per_site=2))
    transfer = np.asarray(transfer)
    rules = {
        "single_first": 0.5 * (cv + transfer[0]),
        "mean_two": 0.5 * (cv + transfer.mean(axis=0)),
        "worst_two": 0.5 * (cv + transfer.min(axis=0)),
        "day1_cv": cv,
        "support_mean_only": transfer.mean(axis=0),
    }
    selected_indices = {name: int(np.argmax(score))
                        for name, score in rules.items()}
    named = {name: tuple(map(int, candidates[index]))
             for name, index in selected_indices.items()}
    named["fixed_eight_site"] = tuple(fixed_grid_design(
        site_coordinates(tuple(meta1["site_names"]))))
    named["fixed_mixed"] = fixed_mixed(meta1)
    named["mean_two_same_contacts"] = physical_contacts(
        named["mean_two"], meta1["pairs"])
    named["worst_two_same_contacts"] = physical_contacts(
        named["worst_two"], meta1["pairs"])
    support = (day2[0][support_ids], day2[1][support_ids])
    query = (day2[0][query_ids], day2[1][query_ids])
    fold = prepare_supported_fold(day1, support, query,
                                  features_per_site=2, support_weight=4.0)
    outcomes = score_programs([fold],
                              np.asarray(list(named.values()), dtype=np.int16),
                              features_per_site=2)
    results = {name: float(score) for name, score in zip(named, outcomes)}
    budget_indices = nested_budget_indices(rules["mean_two"])
    unique_indices = sorted({index for chosen in budget_indices.values()
                             for index in chosen})
    budget_outcomes = score_programs(
        [fold], candidates[unique_indices], features_per_site=2)
    lookup = {index: float(value) for index, value in
              zip(unique_indices, budget_outcomes)}
    budget_curve = {
        str(budget): {
            "selected_indices_by_draw": indices,
            "query_balanced_accuracy_by_draw": [lookup[index]
                                                for index in indices],
            "mean_query_balanced_accuracy": float(np.mean([
                lookup[index] for index in indices])),
            "mean_selection_score": float(np.mean(
                rules["mean_two"][indices])),
        } for budget, indices in budget_indices.items()}
    return {
        "day1_source_record_sha256": meta1["source_record_sha256"],
        "day2_source_record_sha256": meta2["source_record_sha256"],
        "first_support_indices": first_ids.tolist(),
        "second_support_indices": second_ids.tolist(),
        "support_count": len(support_ids), "query_count": len(query_ids),
        "selected": {name: {"program": named[name],
                            "day1_cv": float(cv[index]),
                            "transfer_first": float(transfer[0, index]),
                            "transfer_second": float(transfer[1, index]),
                            "physical_contacts": physical_contacts(
                                named[name], meta1["pairs"])}
                     for name, index in selected_indices.items()},
        "query_balanced_accuracy": results,
        "nested_budget_curve": budget_curve,
        "elapsed_seconds": time.perf_counter() - start,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mono-cache", type=Path, required=True)
    parser.add_argument("--pair-cache", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--candidates", type=int, default=5000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.candidates < max(BUDGETS):
        parser.error("new output and at least 5000 candidates required")
    people = {}
    for subject in args.subjects:
        people[subject] = analyze_subject(
            args.mono_cache, args.pair_cache, subject, args.candidates)
        print(subject, people[subject]["query_balanced_accuracy"], flush=True)
    rules = tuple(people[args.subjects[0]]["query_balanced_accuracy"])
    summary = {}
    for rule in rules:
        values = [person["query_balanced_accuracy"][rule]
                  for person in people.values()]
        fixed = [person["query_balanced_accuracy"]["fixed_eight_site"]
                 for person in people.values()]
        summary[rule] = {"mean_query_balanced_accuracy": float(np.mean(values)),
                         "mean_paired_gain_vs_fixed": float(np.mean(
                             np.asarray(values) - fixed)),
                         "wins_vs_fixed": int(sum(a > b for a, b in
                                                  zip(values, fixed)))}
    result = {
        "schema": "openalterego.hyser-two-support-robust-search.v1",
        "scope": "Retrospective exposed forearm gesture subjects; digitally mixed measured channels, classwise support is not a globally chronological prefix or analog wearable measurement",
        "subjects": people, "candidate_count": args.candidates,
        "candidate_seed": 42, "support_shots_per_class": 2,
        "support_weight": 4.0,
        "physical_cost": "eight array contacts plus common reference; eight AFE channels",
        "selection_rules": {
            "single_first": "half day-one CV plus half first-support transfer",
            "mean_two": "half day-one CV plus half mean of two separate support transfer scores",
            "worst_two": "half day-one CV plus half minimum of two separate support transfer scores",
            "day1_cv": "day-one CV only",
            "support_mean_only": "mean of two support transfer scores",
        },
        "nested_budget_replay": {
            "rule": "mean_two", "candidate_order_seed": 991,
            "draws": BUDGET_DRAWS, "budgets": BUDGETS,
            "meaning": "Ten nested permutations of each same 5000-program pool; all scores precomputed; no independent subjects or runtime speedup claim",
        },
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
