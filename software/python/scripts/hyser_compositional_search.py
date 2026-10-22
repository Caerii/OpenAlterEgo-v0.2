"""Measured mixed-mode edge-program beam search versus matched random search."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from openalterego.research.compositional_search import beam_search, random_search
from openalterego.research.dense_programs import (
    prepare_cv_folds, prepare_fold, score_programs)
from openalterego.research.hyser_features import site_coordinates
from scripts.hyser_mixed_program_pilot import (
    fixed_mixed, load_mixed, physical_contacts)
from scripts.hyser_program_search import fixed_grid_design


def edge_mode_options(meta: dict) -> tuple[tuple[tuple[int, int], ...], ...]:
    """For each physical edge: two monopolar, or either mono plus difference."""
    sites = meta["site_names"]
    grid_ids = {name: index for index, name in
                enumerate(sorted({site[:2] for site in sites}))}
    groups = [[] for _ in range(4)]
    for index, (left, right) in enumerate(meta["pairs"]):
        if sites[left][:2] != sites[right][:2]:
            raise ValueError("pair crosses physical grids")
        grid = grid_ids[sites[left][:2]]
        differential = 256 + index
        groups[grid].extend((tuple(sorted((left, right))),
                             tuple(sorted((left, differential))),
                             tuple(sorted((right, differential)))))
    result = tuple(tuple(sorted(set(group))) for group in groups)
    if any(len(group) != 336 for group in result):
        raise ValueError("expected 336 edge/mode options per grid")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mono-cache", type=Path, required=True)
    parser.add_argument("--pair-cache", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--beam-size", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    result = {"schema": "openalterego.hyser-compositional-search.v1",
              "scope": "Retrospective forearm digital replay; no analog wearable or facial-speech claim",
              "grammar": "one adjacent contact edge per grid; choose two of its three signals (two shared-reference potentials, one digital difference)",
              "physical_cost": "eight array contacts plus one shared reference, eight AFE channels",
              "beam_size": args.beam_size, "random_seed": 42,
              "alpha": 10.0, "subjects": {}}
    for subject in args.subjects:
        train, meta1 = load_mixed(args.mono_cache, args.pair_cache, subject, 1)
        target, meta2 = load_mixed(args.mono_cache, args.pair_cache, subject, 2)
        if meta1["pairs"] != meta2["pairs"] or meta1["site_names"] != meta2["site_names"]:
            raise ValueError(f"pair metadata changed: {subject}")
        options = edge_mode_options(meta1)
        folds = prepare_cv_folds(*train, features_per_site=2)
        score_cv = lambda designs: score_programs(
            folds, designs, features_per_site=2)
        started = time.perf_counter()
        beam = beam_search(options, score_cv, beam_size=args.beam_size)
        beam_seconds = time.perf_counter() - started
        started = time.perf_counter()
        random = random_search(options, score_cv,
                               evaluations=beam.evaluations, seed=42)
        random_seconds = time.perf_counter() - started
        fixed_mono = tuple(fixed_grid_design(
            site_coordinates(tuple(meta1["site_names"]))))
        programs = {"beam": beam.channels, "random": random.channels,
                    "fixed_mixed": fixed_mixed(meta1),
                    "fixed_monopolar": fixed_mono,
                    "beam_same_contacts_monopolar": physical_contacts(
                        beam.channels, meta1["pairs"])}
        fold = prepare_fold(train, target, features_per_site=2)
        names = list(programs)
        actual = score_programs([fold], np.asarray([programs[name] for name in names],
                                                 dtype=np.int16), features_per_site=2)
        result["subjects"][subject] = {
            "day1_source_record_sha256": meta1["source_record_sha256"],
            "day2_source_record_sha256": meta2["source_record_sha256"],
            "beam_evaluations": beam.evaluations,
            "beam_stage_evaluations": beam.stage_evaluations,
            "beam_wall_seconds": beam_seconds,
            "random_wall_seconds": random_seconds,
            "selected_day1_cv": {"beam": beam.score, "random": random.score},
            "choices": {name: {"channels": programs[name],
                               "day2_balanced_accuracy": float(score)}
                        for name, score in zip(names, actual)}}
        print(subject, {name: round(float(score), 3)
                        for name, score in zip(names, actual)},
              f"beam={beam_seconds:.2f}s random={random_seconds:.2f}s",
              flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
