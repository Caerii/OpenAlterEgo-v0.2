"""Search measured digital-differential Hyser pair programs across two days."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from openalterego.research.dense_programs import (
    prepare_cv_folds, prepare_fold, score_programs)
from openalterego.research.hyser_features import site_coordinates


def load(cache: Path, subject: str, session: int):
    paths = list(cache.glob(f"subject{subject}-session{session}-*.npz"))
    if len(paths) != 1:
        raise ValueError(f"expected one differential cache for {subject}/{session}")
    with np.load(paths[0], allow_pickle=False) as data:
        features = data["features"].astype(np.float64)
        labels = data["labels"].astype(np.int64)
    meta = json.loads(paths[0].with_suffix(".json").read_text(encoding="utf-8"))
    if (features.shape != (len(labels), 896) or
            meta["trials"] != len(labels) or len(meta["pairs"]) != 448):
        raise ValueError("invalid differential feature cache")
    return (features, labels), meta


def wedge_options(meta: dict) -> list[list[tuple[int, int]]]:
    """Two pairs sharing a contact, separately within each physical grid."""
    pairs = meta["pairs"]
    points = site_coordinates(tuple(meta["site_names"]))
    incident = [[] for _ in range(256)]
    for index, (left, right) in enumerate(pairs):
        incident[left].append(index)
        incident[right].append(index)
    options = [[] for _ in range(4)]
    for anchor in range(256):
        grid = int(points[anchor, 0])
        channels = incident[anchor]
        for i, left in enumerate(channels):
            for right in channels[i + 1:]:
                options[grid].append(tuple(sorted((left, right))))
    return [sorted(set(group)) for group in options]


def fixed_program(meta: dict) -> tuple[int, ...]:
    points = site_coordinates(tuple(meta["site_names"]))
    lookup = {tuple(map(int, point)): index for index, point in enumerate(points)}
    pair_lookup = {frozenset(pair): index for index, pair in enumerate(meta["pairs"])}
    chosen = []
    for grid in range(4):
        anchor = lookup[(grid, 3, 3)]
        chosen.extend((pair_lookup[frozenset((anchor, lookup[(grid, 3, 4)]))],
                       pair_lookup[frozenset((anchor, lookup[(grid, 4, 3)]))]))
    return tuple(sorted(chosen))


def candidate_programs(meta: dict, count: int, seed: int = 42) -> np.ndarray:
    if count < 1:
        raise ValueError("candidate count must be positive")
    options = wedge_options(meta)
    rng = np.random.default_rng(seed)
    programs = set()
    while len(programs) < count:
        program = tuple(sorted(channel for group in options
                               for channel in group[rng.integers(len(group))]))
        programs.add(program)
    return np.asarray(sorted(programs), dtype=np.int16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--candidates", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    result = {"schema": "openalterego.hyser-differential-program-pilot.v1",
              "scope": "Retrospective digital differences of preprocessed forearm recordings, not analog-front-end or facial wearable evidence",
              "program": "two adjacent differential pairs sharing a contact per grid; eight AFE channels, 12 distinct physical contacts",
              "candidate_count": args.candidates, "seed": 42, "alpha": 10.0,
              "subjects": {}}
    for subject in args.subjects:
        train, meta1 = load(args.cache_dir, subject, 1)
        designs = candidate_programs(meta1, args.candidates)
        fixed = fixed_program(meta1)
        cv = score_programs(prepare_cv_folds(*train, features_per_site=2),
                            designs, features_per_site=2)
        chosen = int(np.argmax(cv))
        target, meta2 = load(args.cache_dir, subject, 2)
        if meta1["pairs"] != meta2["pairs"] or meta1["site_names"] != meta2["site_names"]:
            raise ValueError("pair maps changed between sessions")
        fold = prepare_fold(train, target, features_per_site=2)
        actual = score_programs([fold], designs, features_per_site=2)
        fixed_actual = float(score_programs([fold], np.asarray([fixed]),
                                            features_per_site=2)[0])
        selected_actual = float(actual[chosen])
        result["subjects"][subject] = {
            "day1_source_record_sha256": meta1["source_record_sha256"],
            "day2_source_record_sha256": meta2["source_record_sha256"],
            "fixed_channels": fixed,
            "chosen_channels": designs[chosen].astype(int).tolist(),
            "chosen_day1_cv": float(cv[chosen]),
            "fixed_day2_balanced_accuracy": fixed_actual,
            "chosen_day2_balanced_accuracy": selected_actual,
            "chosen_minus_fixed": selected_actual - fixed_actual,
            "candidate_day2_median": float(np.median(actual)),
            "candidate_day2_oracle": float(actual.max()),
            "day1_cv_day2_spearman": float(spearmanr(cv, actual).statistic)}
        print(subject, "chosen", round(selected_actual, 3), "fixed",
              round(fixed_actual, 3), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
