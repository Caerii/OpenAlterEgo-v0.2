"""Search mixed shared-reference and adjacent-difference programs at nine contacts."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from openalterego.research.dense_programs import (
    prepare_cv_folds, prepare_fold, score_programs)
from openalterego.research.hyser_features import site_coordinates
from scripts.hyser_differential_program_pilot import load as load_pairs
from scripts.hyser_program_search import fixed_grid_design, load_session


def load_mixed(mono_cache: Path, pair_cache: Path, subject: str, session: int):
    mono, mono_meta = load_session(mono_cache, subject, session)
    diff, diff_meta = load_pairs(pair_cache, subject, session)
    if (mono_meta["source_record_sha256"] != diff_meta["source_record_sha256"] or
            mono_meta["site_names"] != diff_meta["site_names"] or
            not np.array_equal(mono[1], diff[1])):
        raise ValueError(f"mixed-mode source or trial alignment differs: {subject}/{session}")
    return (np.concatenate((mono[0], diff[0]), axis=1), mono[1]), diff_meta


def mixed_programs(meta: dict, count: int, seed: int = 42) -> np.ndarray:
    """One anchor monopolar and its local difference per grid."""
    if count < 1:
        raise ValueError("candidate count must be positive")
    sites, pairs = meta["site_names"], meta["pairs"]
    by_grid = [[] for _ in range(4)]
    grid_ids = {name: index for index, name in
                enumerate(sorted({site[:2] for site in sites}))}
    for pair_index, (anchor, neighbor) in enumerate(pairs):
        grid = grid_ids[sites[anchor][:2]]
        if sites[anchor][:2] != sites[neighbor][:2]:
            raise ValueError("pair crosses physical grids")
        by_grid[grid].append((anchor, 256 + pair_index))
    rng = np.random.default_rng(seed)
    programs = set()
    while len(programs) < count:
        programs.add(tuple(sorted(channel for options in by_grid
                                  for channel in options[rng.integers(len(options))])))
    return np.asarray(sorted(programs), dtype=np.int16)


def fixed_mixed(meta: dict) -> tuple[int, ...]:
    sites = tuple(meta["site_names"])
    points = site_coordinates(sites)
    lookup = {tuple(map(int, point)): index for index, point in enumerate(points)}
    pair_lookup = {frozenset(pair): index for index, pair in enumerate(meta["pairs"])}
    channels = []
    for grid in range(4):
        anchor = lookup[(grid, 3, 3)]
        neighbor = lookup[(grid, 3, 4)]
        channels.extend((anchor, 256 + pair_lookup[frozenset((anchor, neighbor))]))
    return tuple(sorted(channels))


def physical_contacts(program: tuple[int, ...] | np.ndarray,
                      pairs: list[list[int]]) -> tuple[int, ...]:
    contacts = set()
    for channel in program:
        channel = int(channel)
        if channel < 256:
            contacts.add(channel)
        else:
            contacts.update(pairs[channel - 256])
    if len(contacts) != 8:
        raise ValueError("mixed program must use eight array contacts")
    return tuple(sorted(contacts))


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
    result = {"schema": "openalterego.hyser-mixed-program-pilot.v1",
              "scope": "Retrospective measured forearm replay; digital differences omit analog contact and AFE behavior",
              "program": "one shared-reference contact and one adjacent difference sharing that contact on each grid; eight channels, eight array contacts plus common reference",
              "candidate_count": args.candidates, "seed": 42,
              "alpha": 10.0, "subjects": {}}
    for subject in args.subjects:
        train, meta1 = load_mixed(args.mono_cache, args.pair_cache, subject, 1)
        target, meta2 = load_mixed(args.mono_cache, args.pair_cache, subject, 2)
        if meta1["pairs"] != meta2["pairs"] or meta1["site_names"] != meta2["site_names"]:
            raise ValueError(f"pair metadata changed: {subject}")
        designs = mixed_programs(meta1, args.candidates)
        cv = score_programs(prepare_cv_folds(*train, features_per_site=2),
                            designs, features_per_site=2)
        selected = int(np.argmax(cv))
        fold = prepare_fold(train, target, features_per_site=2)
        actual = score_programs([fold], designs, features_per_site=2)
        fixed_mix = fixed_mixed(meta1)
        fixed_mono = fixed_grid_design(site_coordinates(tuple(meta1["site_names"])))
        selected_contacts = physical_contacts(designs[selected], meta1["pairs"])
        fixed_mix_contacts = physical_contacts(fixed_mix, meta1["pairs"])
        comparators = score_programs([fold], np.asarray([
            fixed_mix, fixed_mono, selected_contacts, fixed_mix_contacts]),
                                     features_per_site=2)
        result["subjects"][subject] = {
            "day1_source_record_sha256": meta1["source_record_sha256"],
            "day2_source_record_sha256": meta2["source_record_sha256"],
            "selected_channels": designs[selected].astype(int).tolist(),
            "selected_day1_cv": float(cv[selected]),
            "selected_day2": float(actual[selected]),
            "fixed_mixed_day2": float(comparators[0]),
            "fixed_monopolar_day2": float(comparators[1]),
            "selected_same_contacts_monopolar_day2": float(comparators[2]),
            "fixed_mixed_same_contacts_monopolar_day2": float(comparators[3]),
            "selected_array_contacts": selected_contacts,
            "candidate_day2_median": float(np.median(actual)),
            "candidate_day2_oracle": float(actual.max()),
            "day1_cv_day2_spearman": float(spearmanr(cv, actual).statistic)}
        print(subject, "selected", round(float(actual[selected]), 3),
              "mono", round(float(comparators[1]), 3), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
