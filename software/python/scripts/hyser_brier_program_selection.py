"""Compare proper-score and hard-accuracy selection on identical programs."""

import argparse
import json
from pathlib import Path

import numpy as np

from openalterego.research.dense_programs import prepare_cv_folds, prepare_fold, score_programs
from openalterego.research.hyser_features import site_coordinates
from openalterego.research.supported_programs import prepare_supported_fold
from scripts.hyser_compositional_search import edge_mode_options
from scripts.hyser_mixed_program_pilot import load_mixed, physical_contacts
from scripts.hyser_program_search import fixed_grid_design
from scripts.hyser_support_conditioned_search import sample_programs
from scripts.hyser_two_support_robust_search import two_support_split


def analyze_subject(mono_cache: Path, pair_cache: Path, subject: str,
                    candidates: int) -> dict:
    day1, meta1 = load_mixed(mono_cache, pair_cache, subject, 1)
    day2, meta2 = load_mixed(mono_cache, pair_cache, subject, 2)
    if (meta1["site_names"] != meta2["site_names"] or
            meta1["pairs"] != meta2["pairs"] or
            meta1["source_record_sha256"] == meta2["source_record_sha256"]):
        raise ValueError("day pair does not align")
    first, second, support_ids, query_ids = two_support_split(day2[1])
    programs = sample_programs(edge_mode_options(meta1), candidates, 42)
    cv = prepare_cv_folds(*day1, features_per_site=2)
    transfer = [prepare_fold(day1, (day2[0][ids], day2[1][ids]),
                             features_per_site=2) for ids in (first, second)]
    selection = {}
    for metric in ("accuracy", "brier"):
        cv_scores = score_programs(cv, programs, features_per_site=2,
                                   metric=metric)
        transfers = [score_programs([fold], programs, features_per_site=2,
                                    metric=metric) for fold in transfer]
        selection[metric] = .5 * cv_scores + .25 * (transfers[0] + transfers[1])
    chosen = {metric: int(np.argmax(scores))
              for metric, scores in selection.items()}
    named = {metric: tuple(map(int, programs[index]))
             for metric, index in chosen.items()}
    named["fixed_eight_site"] = tuple(fixed_grid_design(
        site_coordinates(tuple(meta1["site_names"]))))
    fold = prepare_supported_fold(
        day1, (day2[0][support_ids], day2[1][support_ids]),
        (day2[0][query_ids], day2[1][query_ids]),
        features_per_site=2, support_weight=4.0)
    outcomes = score_programs([fold], np.asarray(list(named.values()),
                                                 dtype=np.int16),
                              features_per_site=2)
    return {
        "day1_source_record_sha256": meta1["source_record_sha256"],
        "day2_source_record_sha256": meta2["source_record_sha256"],
        "support_indices": support_ids.tolist(), "query_indices": query_ids.tolist(),
        "chosen": {metric: {"index": chosen[metric],
                            "program": named[metric],
                            "physical_contacts": physical_contacts(
                                named[metric], meta1["pairs"]),
                            "selection_score": float(selection[metric][chosen[metric]])}
                   for metric in chosen},
        "query_balanced_accuracy": {name: float(value)
                                    for name, value in zip(named, outcomes)},
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
        "schema": "openalterego.hyser-brier-program-selection.v1",
        "scope": "Retrospective exposed forearm subjects; digitally derived signals only",
        "selection": "Identical seed-42 complete candidate pool, half day-one CV plus one-quarter each independent support repetition; raw ridge outputs softmaxed at temperature one for class-balanced Brier score",
        "candidate_count": args.candidates,
        "subjects": people,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
