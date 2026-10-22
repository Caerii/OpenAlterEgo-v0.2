"""Matched four-support refit after optional extra-repetition program selection."""

import argparse
import json
from pathlib import Path

import numpy as np

from openalterego.research.dense_programs import prepare_cv_folds, prepare_fold, score_programs
from openalterego.research.hyser_features import site_coordinates
from openalterego.research.supported_programs import prepare_supported_fold
from scripts.hyser_compositional_search import edge_mode_options
from scripts.hyser_mixed_program_pilot import load_mixed, physical_contacts
from scripts.hyser_mixed_rank_transport import split_remaining_repetitions
from scripts.hyser_program_search import fixed_grid_design
from scripts.hyser_support_conditioned_search import sample_programs
from scripts.hyser_two_support_robust_search import two_support_split


def analyze_subject(mono_cache: Path, pair_cache: Path, subject: str,
                    candidate_count: int) -> dict:
    day1, meta1 = load_mixed(mono_cache, pair_cache, subject, 1)
    day2, meta2 = load_mixed(mono_cache, pair_cache, subject, 2)
    if (meta1["site_names"] != meta2["site_names"] or
            meta1["pairs"] != meta2["pairs"] or
            meta1["source_record_sha256"] == meta2["source_record_sha256"]):
        raise ValueError("day pair does not align")
    first, second, initial_support, remaining = two_support_split(day2[1])
    extra_support, query, excluded = split_remaining_repetitions(day2[1], remaining)
    all_support = np.concatenate((initial_support, extra_support))
    if set(all_support) & set(query):
        raise ValueError("support and evaluation overlap")
    programs = sample_programs(edge_mode_options(meta1), candidate_count, 42)
    cv = score_programs(prepare_cv_folds(*day1, features_per_site=2),
                        programs, features_per_site=2)
    transfers = [score_programs(
        [prepare_fold(day1, (day2[0][ids], day2[1][ids]), features_per_site=2)],
        programs, features_per_site=2) for ids in (first, second)]
    initial_scores = .5 * cv + .25 * (transfers[0] + transfers[1])
    initial_choice = int(np.argmax(initial_scores))
    extra_fold = prepare_supported_fold(
        day1, (day2[0][initial_support], day2[1][initial_support]),
        (day2[0][extra_support], day2[1][extra_support]),
        features_per_site=2, support_weight=4.0)
    extra_scores = score_programs([extra_fold], programs, features_per_site=2)
    extra_choice = int(np.argmax(extra_scores))
    fixed = tuple(fixed_grid_design(
        site_coordinates(tuple(meta1["site_names"]))))
    named = {"two_support_selector": programs[initial_choice],
             "four_support_selector": programs[extra_choice],
             "fixed_eight_site": fixed}
    final_fold = prepare_supported_fold(
        day1, (day2[0][all_support], day2[1][all_support]),
        (day2[0][query], day2[1][query]),
        features_per_site=2, support_weight=4.0)
    outcomes = score_programs([final_fold], np.asarray(list(named.values()),
                                                 dtype=np.int16),
                              features_per_site=2)
    return {
        "day1_source_record_sha256": meta1["source_record_sha256"],
        "day2_source_record_sha256": meta2["source_record_sha256"],
        "initial_support_indices": initial_support.tolist(),
        "extra_support_indices": extra_support.tolist(),
        "query_indices": query.tolist(),
        "excluded_from_partition_indices": excluded.tolist(),
        "choices": {
            name: {"program": list(map(int, program)),
                   "physical_contacts": physical_contacts(program, meta1["pairs"])}
            for name, program in named.items()},
        "selection_scores": {
            "initial": float(initial_scores[initial_choice]),
            "extra": float(extra_scores[extra_choice])},
        "query_balanced_accuracy": {
            name: float(score) for name, score in zip(named, outcomes)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mono-cache", type=Path, required=True)
    parser.add_argument("--pair-cache", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--candidates", type=int, default=5000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.candidates < 1:
        parser.error("new output and positive candidate count required")
    people = {}
    for subject in args.subjects:
        people[subject] = analyze_subject(args.mono_cache, args.pair_cache,
                                          subject, args.candidates)
        print(subject, people[subject]["query_balanced_accuracy"], flush=True)
    result = {
        "schema": "openalterego.hyser-four-support-selection.v1",
        "scope": "Retrospective exposed forearm participants; within-day disjoint B query, not a new session or hardware intervention",
        "candidate_count": args.candidates, "candidate_seed": 42,
        "selection": "initial day-one CV plus two transfer repetitions versus accuracy on extra A repetitions after two-support decoder fit; both finalists and fixed refit with all four support repetitions before B evaluation",
        "subjects": people,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
