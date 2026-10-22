"""Search measured designs for tolerance of one-cell session displacement."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from openalterego.research.contact_shift import (
    SHIFTS, align_shifted_features, channel_shift_maps, robust_options)
from openalterego.research.dense_programs import (
    prepare_cv_folds, prepare_fold, score_programs)
from openalterego.research.hyser_features import site_coordinates
from openalterego.research.hyser_fewshot import support_query
from openalterego.research.supported_programs import prepare_supported_fold
from scripts.hyser_compositional_search import edge_mode_options
from scripts.hyser_mixed_program_pilot import load_mixed
from scripts.hyser_program_search import fixed_grid_design
from scripts.hyser_support_conditioned_search import sample_programs


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
        "schema": "openalterego.hyser-contact-shift-search.v1",
        "scope": "Retrospective dense-array forearm replay, one-cell topological displacement from day one to day two",
        "shifts": SHIFTS, "candidate_seed": 42,
        "candidate_count": args.candidates,
        "selection": "nominal = half day1 CV, half nominal support transfer; robust = maximum worst shifted half-CV-half-transfer",
        "query": "day1 nominal plus one labeled day2 support per class at shifted sites, weight 4; remaining shifted day2 trials query",
        "subjects": {},
    }
    for subject in args.subjects:
        started = time.perf_counter()
        day1, meta1 = load_mixed(args.mono_cache, args.pair_cache, subject, 1)
        day2, meta2 = load_mixed(args.mono_cache, args.pair_cache, subject, 2)
        if (meta1["site_names"] != meta2["site_names"] or
                meta1["pairs"] != meta2["pairs"]):
            raise ValueError(f"session metadata differs: {subject}")
        maps = channel_shift_maps(meta1)
        options = robust_options(edge_mode_options(meta1), maps)
        candidates = sample_programs(options, args.candidates, 42)
        fixed = tuple(fixed_grid_design(
            site_coordinates(tuple(meta1["site_names"]))))
        if np.any(maps[:, fixed] < 0):
            raise ValueError("fixed layout crosses shift boundary")
        support_ids, query_ids = support_query(day2[1], 1)
        cv = score_programs(prepare_cv_folds(*day1, features_per_site=2),
                            candidates, features_per_site=2)
        transfer = []
        shifted_days = []
        for channel_map in maps:
            x2 = align_shifted_features(day2[0], channel_map,
                                        features_per_channel=2)
            shifted_days.append((x2, day2[1]))
            support = (x2[support_ids], day2[1][support_ids])
            transfer.append(score_programs(
                [prepare_fold(day1, support, features_per_site=2)],
                candidates, features_per_site=2))
        transfer = np.asarray(transfer)
        objective = (cv[None, :] + transfer) / 2
        nominal_index = int(np.argmax(objective[0]))
        robust_index = int(np.argmax(np.min(objective, axis=0)))
        programs = {
            "nominal_selected": tuple(map(int, candidates[nominal_index])),
            "worst_shift_selected": tuple(map(int, candidates[robust_index])),
            "fixed_eight_site": fixed,
        }
        scores = {}
        for shift, shifted_day in zip(SHIFTS, shifted_days):
            x2, y2 = shifted_day
            fold = prepare_supported_fold(
                day1, (x2[support_ids], y2[support_ids]),
                (x2[query_ids], y2[query_ids]),
                features_per_site=2, support_weight=4.0)
            values = score_programs(
                [fold], np.asarray(list(programs.values()), dtype=np.int16),
                features_per_site=2)
            scores[str(shift)] = {name: float(value)
                                  for name, value in zip(programs, values)}
        result["subjects"][subject] = {
            "day1_source_record_sha256": meta1["source_record_sha256"],
            "day2_source_record_sha256": meta2["source_record_sha256"],
            "grammar_options_per_grid": tuple(len(group) for group in options),
            "support_trials": len(support_ids), "query_trials": len(query_ids),
            "programs": programs,
            "selected_nominal_objective": float(objective[0, nominal_index]),
            "selected_worst_shift_objective": float(
                np.min(objective[:, robust_index])),
            "query_balanced_accuracy_by_shift": scores,
            "elapsed_seconds": time.perf_counter() - started,
        }
        print(subject, {name: round(float(np.mean([scores[str(shift)][name]
                                                  for shift in SHIFTS])), 3)
                        for name in programs}, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
