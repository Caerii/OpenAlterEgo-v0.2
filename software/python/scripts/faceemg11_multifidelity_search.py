"""Screen facial programs cheaply, then rerank finalists with temporal features."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from openalterego.research.dense_programs import (
    prepare_cv_folds, prepare_fold, score_programs)
from scripts.faceemg11_program_search import (
    CANDIDATES, load_subject, split_subject)


def temporal_subject(cache: Path, subject: str, source_hash: str):
    stem = f"sub-{subject}_trials-temporal"
    metadata = json.loads((cache / (stem + ".json")).read_text(encoding="utf-8"))
    if (metadata.get("schema") != "openalterego.faceemg11-four-bin.v1" or
            metadata.get("source_member_sha256") != source_hash):
        raise ValueError(f"source identity differs: {subject}")
    with np.load(cache / (stem + ".npz"), allow_pickle=False) as data:
        features = data["features"].astype(np.float64)
        labels = data["labels"].copy()
        groups = data["groups"].copy()
    if features.shape != (330, 160) or not np.isfinite(features).all():
        raise ValueError(f"unexpected temporal feature matrix: {subject}")
    return features, labels, groups


def finalists(scores: np.ndarray, count: int) -> np.ndarray:
    """Stable top-K indices; candidate tuples are already lexicographic."""
    if scores.shape != (len(CANDIDATES),) or not 1 <= count <= len(scores):
        raise ValueError("invalid candidate scores or finalist count")
    return np.argsort(-scores, kind="stable")[:count]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse-cache", type=Path, required=True)
    parser.add_argument("--temporal-cache", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--finalists", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    result = {
        "schema": "openalterego.faceemg11-multifidelity-search.v1",
        "scope": "Post hoc later-block analysis; subjects 07-12 query outcomes previously exposed",
        "source": "https://doi.org/10.5281/zenodo.22137268",
        "low_feature_count_per_channel": 2,
        "high_feature_count_per_channel": 8,
        "candidate_count": len(CANDIDATES),
        "finalist_count": args.finalists,
        "selection": "low three-fold training CV top-K; high three-fold training CV reranks K",
        "query": "blocks 21-30; high-feature model fitted blocks 1-20",
        "subjects": {},
    }
    for subject in args.subjects:
        coarse_x, coarse_y, coarse_groups, metadata = load_subject(
            args.coarse_cache, subject)
        temporal_x, temporal_y, temporal_groups = temporal_subject(
            args.temporal_cache, subject, metadata["member_sha256"])
        if (not np.array_equal(coarse_y, temporal_y) or
                not np.array_equal(coarse_groups, temporal_groups)):
            raise ValueError(f"trial alignment differs: {subject}")
        coarse_train, _, _ = split_subject(coarse_x, coarse_y, coarse_groups)
        temporal_train, temporal_validation, temporal_query = split_subject(
            temporal_x, temporal_y, temporal_groups)
        started = time.perf_counter()
        low_scores = score_programs(
            prepare_cv_folds(*coarse_train, features_per_site=2),
            CANDIDATES, features_per_site=2)
        low_seconds = time.perf_counter() - started
        shortlist = finalists(low_scores, args.finalists)
        started = time.perf_counter()
        high_scores = score_programs(
            prepare_cv_folds(*temporal_train, features_per_site=8),
            CANDIDATES[shortlist], features_per_site=8)
        high_seconds = time.perf_counter() - started
        low_index = int(shortlist[0])
        high_index = int(shortlist[np.argmax(high_scores)])
        programs = {
            "low_feature_selected": tuple(map(int, CANDIDATES[low_index])),
            "temporal_reranked": tuple(map(int, CANDIDATES[high_index])),
        }
        fit = (np.vstack((temporal_train[0], temporal_validation[0])),
               np.r_[temporal_train[1], temporal_validation[1]])
        fold = prepare_fold(fit, temporal_query, features_per_site=8)
        values = score_programs(
            [fold], np.asarray(list(programs.values()), dtype=np.int16),
            features_per_site=8)
        result["subjects"][subject] = {
            "source_member_sha256": metadata["member_sha256"],
            "programs": programs,
            "low_selection_cv": float(low_scores[low_index]),
            "high_rerank_cv": float(high_scores.max()),
            "same_program": low_index == high_index,
            "query_balanced_accuracy": {
                name: float(value) for name, value in zip(programs, values)},
            "low_screen_seconds": low_seconds,
            "high_rerank_seconds": high_seconds,
        }
        print(subject, result["subjects"][subject]["query_balanced_accuracy"],
              f"screen={low_seconds:.2f}s rerank={high_seconds:.2f}s", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
