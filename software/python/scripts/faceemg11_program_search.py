"""Exhaustive six-channel facial program search with forward block evaluation."""

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np

from openalterego.research.dense_programs import (
    prepare_cv_folds, prepare_fold, score_programs)


CANDIDATES = np.asarray(list(itertools.combinations(range(20), 6)),
                        dtype=np.int16)
FIXED = (0, 4, 8, 12, 16, 19)


def load_subject(cache: Path, subject: str):
    stem = f"sub-{subject}_trials-features"
    metadata = json.loads((cache / (stem + ".json")).read_text(encoding="utf-8"))
    if (metadata.get("schema") != "openalterego.faceemg11-features.v1" or
            metadata.get("shape") != [330, 40]):
        raise ValueError(f"unexpected facial feature metadata: {subject}")
    with np.load(cache / (stem + ".npz"), allow_pickle=False) as data:
        features = data["features"].astype(np.float64)
        labels = data["labels"].copy()
        groups = data["groups"].copy()
    if (features.shape != (330, 40) or labels.shape != (330,) or
            groups.shape != (330,) or not np.isfinite(features).all()):
        raise ValueError(f"unexpected facial feature cache: {subject}")
    return features, labels, groups, metadata


def split_subject(features, labels, groups):
    train = groups <= 15
    validation = (groups >= 16) & (groups <= 20)
    query = groups >= 21
    if (sum(train) != 165 or sum(validation) != 55 or sum(query) != 110 or
            not np.array_equal(np.unique(labels[train], return_counts=True)[1],
                               np.full(11, 15))):
        raise ValueError("unexpected chronological block split")
    return ((features[train], labels[train]),
            (features[validation], labels[validation]),
            (features[query], labels[query]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    result = {
        "schema": "openalterego.faceemg11-program-search.v1",
        "source": "https://doi.org/10.5281/zenodo.22137268",
        "scope": "Facial and ocular commands, not silent speech; no measured electrode geometry or cross-session reseating",
        "candidate_count": len(CANDIDATES),
        "contact_budget": 6,
        "split": "blocks 1-15 fit, 16-20 selection, 21-30 query; refit on blocks 1-20",
        "fixed_channels": FIXED, "subjects": {},
    }
    for subject in args.subjects:
        started = time.perf_counter()
        features, labels, groups, metadata = load_subject(args.feature_cache, subject)
        train, validation, query = split_subject(features, labels, groups)
        cv = score_programs(prepare_cv_folds(*train, features_per_site=2),
                            CANDIDATES, features_per_site=2)
        forward = score_programs(
            [prepare_fold(train, validation, features_per_site=2)],
            CANDIDATES, features_per_site=2)
        cv_index = int(np.argmax(cv))
        forward_index = int(np.argmax(forward))
        programs = {
            "train_cv_selected": tuple(map(int, CANDIDATES[cv_index])),
            "forward_selected": tuple(map(int, CANDIDATES[forward_index])),
            "fixed_six_index_spread": FIXED,
        }
        x_fit = np.vstack((train[0], validation[0]))
        y_fit = np.r_[train[1], validation[1]]
        fold = prepare_fold((x_fit, y_fit), query, features_per_site=2)
        values = score_programs(
            [fold], np.asarray(list(programs.values()), dtype=np.int16),
            features_per_site=2)
        all_twenty = score_programs(
            [fold], np.arange(20, dtype=np.int16)[None, :],
            features_per_site=2)[0]
        result["subjects"][subject] = {
            "source_member_sha256": metadata["member_sha256"],
            "programs": programs,
            "selected_train_cv": float(cv[cv_index]),
            "selected_forward_validation": float(forward[forward_index]),
            "query_balanced_accuracy": {
                name: float(value) for name, value in zip(programs, values)
            } | {"all_twenty_channel_reference": float(all_twenty)},
            "elapsed_seconds": time.perf_counter() - started,
        }
        print(subject, {name: round(value, 3) for name, value in
                        result["subjects"][subject]["query_balanced_accuracy"].items()},
              flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
