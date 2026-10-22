"""Build day-one predictors and next-day labels for grouped layout learning."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from openalterego.research.hyser_features import site_coordinates
from openalterego.research.hyser_search import prepare_folds
from openalterego.research.hyser_site_health import dark_contact_mask
from openalterego.research.hyser_spatial_search import (
    candidate_designs, displacement_maps, displacement_scores)
from openalterego.research.hyser_transfer import prepare_day_pair, score_day_pair
from scripts.hyser_program_search import fixed_grid_design, load_session


FEATURE_NAMES = ["nominal_cv", "mean_shift_cv", "minimum_shift_cv",
                 "shift_cv_sd", "mean_darkness", "max_darkness",
                 "dark_contact_fraction", "mean_log_rms", "mean_log_rms_sd",
                 "mean_trial_feature_sd", "min_pair_grid_cells",
                 "mean_pair_grid_cells", "day_one_trials", "day_one_classes"]


def day_one_predictors(train, sites: np.ndarray, designs: np.ndarray) -> np.ndarray:
    x, labels = train
    scores = displacement_scores(prepare_folds(x, labels), designs,
                                 displacement_maps(sites))
    _, darkness = dark_contact_mask(x, sites)
    rms = np.median(x[:, ::2], axis=0)
    spread = x[:, ::2].std(axis=0)
    distances = []
    for design in designs:
        coords = sites[design]
        pairs = [float(np.linalg.norm(coords[i, 1:] - coords[j, 1:]))
                 for i in range(len(design)) for j in range(i + 1, len(design))
                 if coords[i, 0] == coords[j, 0]]
        distances.append((min(pairs), float(np.mean(pairs))))
    distances = np.asarray(distances)
    return np.column_stack((
        scores[:, 0], scores.mean(axis=1), scores.min(axis=1),
        scores.std(axis=1), darkness[designs].mean(axis=1),
        darkness[designs].max(axis=1), (darkness[designs] > 0.5).mean(axis=1),
        rms[designs].mean(axis=1), rms[designs].std(axis=1),
        spread[designs].mean(axis=1), distances,
        np.full(len(designs), len(x)), np.full(len(designs), len(np.unique(labels)))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--candidates", type=int, default=1000)
    args = parser.parse_args()
    if args.output.exists() or args.metadata.exists():
        parser.error("output already exists")
    features, targets, source_hashes = [], [], {}
    designs = None
    expected_sites = None
    for subject in args.subjects:
        train, train_meta = load_session(args.cache_dir, subject, 1)
        target, target_meta = load_session(args.cache_dir, subject, 2)
        if expected_sites is None:
            expected_sites = train_meta["site_names"]
            sites = site_coordinates(tuple(expected_sites))
            random = candidate_designs(sites, count=args.candidates)
            designs = np.vstack((random, fixed_grid_design(sites))).astype(np.int16)
        elif train_meta["site_names"] != expected_sites:
            raise ValueError("site order differs across subjects")
        features.append(day_one_predictors(train, sites, designs))
        targets.append(score_day_pair(prepare_day_pair(train, target), designs))
        source_hashes[subject] = [train_meta["source_record_sha256"],
                                  target_meta["source_record_sha256"]]
        print(subject, "features", features[-1].shape,
              "fixed outcome", round(float(targets[-1][-1]), 3), flush=True)
    features = np.asarray(features)
    targets = np.asarray(targets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        np.savez_compressed(stream, predictors=features, targets=targets,
                            designs=designs, subjects=np.asarray(args.subjects))
    metadata = {"schema": "openalterego.hyser-meta-dataset.v1",
                "scope": "Development data; all next-day outcomes on these subjects have been seen",
                "feature_names": FEATURE_NAMES, "subjects": args.subjects,
                "candidate_count": args.candidates,
                "last_design": "fixed_grid", "seed": 42,
                "source_record_sha256": source_hashes,
                "npz_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "shape": list(features.shape)}
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    with args.metadata.open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
