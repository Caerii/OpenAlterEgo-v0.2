"""Nested large-pool search and later-day selection-overfit diagnostic."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from openalterego.research.dense_programs import prepare_cv_folds, prepare_fold, score_programs
from openalterego.research.hyser_features import site_coordinates
from openalterego.research.supported_programs import prepare_supported_fold
from openalterego.research.uniform_program_pool import uniform_program_prefix
from scripts.hyser_compositional_search import edge_mode_options
from scripts.hyser_mixed_program_pilot import load_mixed, physical_contacts
from scripts.hyser_program_search import fixed_grid_design
from scripts.hyser_two_support_robust_search import two_support_split


def analyze_subject(mono_cache: Path, pair_cache: Path, subject: str,
                    budgets: tuple[int, ...]) -> dict:
    day1, meta1 = load_mixed(mono_cache, pair_cache, subject, 1)
    day2, meta2 = load_mixed(mono_cache, pair_cache, subject, 2)
    if (meta1["site_names"] != meta2["site_names"] or
            meta1["pairs"] != meta2["pairs"] or
            meta1["source_record_sha256"] == meta2["source_record_sha256"]):
        raise ValueError("day pair does not align")
    first, second, support_ids, query_ids = two_support_split(day2[1])
    start = time.perf_counter()
    programs = uniform_program_prefix(edge_mode_options(meta1), max(budgets), 42)
    sampling_seconds = time.perf_counter() - start
    cv_folds = prepare_cv_folds(*day1, features_per_site=2)
    transfer_folds = [prepare_fold(day1, (day2[0][ids], day2[1][ids]),
                                   features_per_site=2)
                      for ids in (first, second)]
    start = time.perf_counter()
    cv = score_programs(cv_folds, programs, features_per_site=2)
    cv_seconds = time.perf_counter() - start
    transfers = []
    transfer_seconds = []
    for fold in transfer_folds:
        start = time.perf_counter()
        transfers.append(score_programs([fold], programs, features_per_site=2))
        transfer_seconds.append(time.perf_counter() - start)
    selection = .5 * cv + .25 * (transfers[0] + transfers[1])
    choices = {budget: int(np.argmax(selection[:budget])) for budget in budgets}
    named = {str(budget): programs[index] for budget, index in choices.items()}
    named["fixed_eight_site"] = tuple(fixed_grid_design(
        site_coordinates(tuple(meta1["site_names"]))))
    fold = prepare_supported_fold(
        day1, (day2[0][support_ids], day2[1][support_ids]),
        (day2[0][query_ids], day2[1][query_ids]),
        features_per_site=2, support_weight=4.0)
    query = score_programs([fold], np.asarray(list(named.values()),
                                             dtype=np.int16),
                           features_per_site=2)
    return {
        "day1_source_record_sha256": meta1["source_record_sha256"],
        "day2_source_record_sha256": meta2["source_record_sha256"],
        "support_indices": support_ids.tolist(),
        "query_indices": query_ids.tolist(),
        "times_seconds": {"sampling": sampling_seconds,
                          "day1_cv": cv_seconds,
                          "support_transfers": transfer_seconds},
        "choices": {name: {"program": list(map(int, program)),
                           "physical_contacts": physical_contacts(
                               program, meta1["pairs"]),
                           "selection_score": float(selection[choices[int(name)]])
                           if name != "fixed_eight_site" else None,
                           "query_balanced_accuracy": float(query[i])}
                    for i, (name, program) in enumerate(named.items())},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mono-cache", type=Path, required=True)
    parser.add_argument("--pair-cache", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--budgets", type=int, nargs="+",
                        default=[5000, 20000, 100000])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    budgets = tuple(args.budgets)
    if (args.output.exists() or not budgets or any(n < 1 for n in budgets) or
            tuple(sorted(set(budgets))) != budgets):
        parser.error("new output and unique increasing positive budgets required")
    people = {}
    for subject in args.subjects:
        people[subject] = analyze_subject(args.mono_cache, args.pair_cache,
                                          subject, budgets)
        print(subject, {name: round(row["query_balanced_accuracy"], 3)
                        for name, row in people[subject]["choices"].items()},
              flush=True)
    result = {
        "schema": "openalterego.hyser-candidate-scale-audit.v1",
        "scope": "Retrospective exposed forearm digital replay; later-day query labels used only after selections",
        "budgets": budgets, "candidate_seed": 42,
        "selection": "half day-one CV plus one-quarter each of two separate support transfer accuracies",
        "subjects": people,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
