"""Development replay: select physical programs using labeled session support."""

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


def sample_programs(options, count: int, seed: int) -> np.ndarray:
    """Unique complete grammar programs with reproducible uniform part draws."""
    if count < 1:
        raise ValueError("candidate count must be positive")
    rng = np.random.default_rng(seed)
    programs = set()
    while len(programs) < count:
        selected = tuple(sorted(channel for group in options
                                for channel in group[rng.integers(len(group))]))
        if len(set(selected)) == 8:
            programs.add(selected)
    return np.asarray(sorted(programs), dtype=np.int16)


def chosen(scores: np.ndarray) -> int:
    """Deterministic tie break by lexicographically sorted program."""
    return int(np.argmax(scores))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mono-cache", type=Path, required=True)
    parser.add_argument("--pair-cache", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--candidates", type=int, default=5000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    result = {
        "schema": "openalterego.hyser-support-conditioned-search.v1",
        "scope": "Exposed development subjects; forearm digital replay, not facial or analog hardware",
        "candidate_count": args.candidates,
        "candidate_seed": 42,
        "support_shots_per_class": 1,
        "query_support_weight": 4.0,
        "selection_rules": ("day1_cv", "day1_to_support", "half_cv_half_support"),
        "physical_cost": "eight array contacts plus common reference; eight AFE channels",
        "subjects": {},
    }
    for subject in args.subjects:
        started = time.perf_counter()
        day1, meta1 = load_mixed(args.mono_cache, args.pair_cache, subject, 1)
        day2, meta2 = load_mixed(args.mono_cache, args.pair_cache, subject, 2)
        if (meta1["source_record_sha256"] == meta2["source_record_sha256"] or
                meta1["site_names"] != meta2["site_names"] or
                meta1["pairs"] != meta2["pairs"]):
            raise ValueError(f"invalid session alignment: {subject}")
        support_ids, query_ids = support_query(day2[1], 1)
        support = (day2[0][support_ids], day2[1][support_ids])
        query = (day2[0][query_ids], day2[1][query_ids])
        candidates = sample_programs(edge_mode_options(meta1), args.candidates, 42)
        cv = score_programs(prepare_cv_folds(*day1, features_per_site=2),
                            candidates, features_per_site=2)
        transfer = score_programs(
            [prepare_fold(day1, support, features_per_site=2)],
            candidates, features_per_site=2)
        rules = {"day1_cv": cv, "day1_to_support": transfer,
                 "half_cv_half_support": (cv + transfer) / 2}
        indices = {name: chosen(scores)
                   for name, scores in rules.items()}
        fixed_mono = tuple(fixed_grid_design(
            site_coordinates(tuple(meta1["site_names"]))))
        fixed_mix = fixed_mixed(meta1)
        named = {name: tuple(map(int, candidates[index]))
                 for name, index in indices.items()}
        named.update((("fixed_eight_site", fixed_mono),
                      ("fixed_mixed", fixed_mix)))
        for name in tuple(indices):
            named[name + "_same_contacts"] = physical_contacts(
                named[name], meta1["pairs"])
        fold = prepare_supported_fold(day1, support, query,
                                      features_per_site=2, support_weight=4.0)
        query_scores = score_programs(
            [fold], np.asarray(list(named.values()), dtype=np.int16),
            features_per_site=2)
        result["subjects"][subject] = {
            "day1_source_record_sha256": meta1["source_record_sha256"],
            "day2_source_record_sha256": meta2["source_record_sha256"],
            "support_trials": len(support_ids), "query_trials": len(query_ids),
            "selection": {name: {"channels": named[name],
                                 "day1_cv": float(cv[index]),
                                 "day1_to_support": float(transfer[index])}
                          for name, index in indices.items()},
            "query_balanced_accuracy": {
                name: float(score) for name, score in zip(named, query_scores)},
            "elapsed_seconds": time.perf_counter() - started,
        }
        print(subject, {name: round(float(score), 3)
                        for name, score in zip(named, query_scores)},
              flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
