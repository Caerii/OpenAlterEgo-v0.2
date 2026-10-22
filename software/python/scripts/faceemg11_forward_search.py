"""Compare shuffled and forward-block selection of sparse facial programs.

This is a retrospective method diagnostic: all later blocks on these subjects
were exposed by prior experiments. It cannot establish a new holdout result.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from openalterego.research.compositional_search import beam_search
from openalterego.research.dense_programs import (
    prepare_cv_folds, prepare_fold, score_programs)
from scripts.faceemg11_multifidelity_search import temporal_subject
from scripts.faceemg11_program_search import load_subject, split_subject


OPTIONS = tuple(tuple((channel,) for channel in range(20)) for _ in range(6))


def forward_folds(features, labels, blocks):
    """Use only earlier blocks to fit each later-block selection fold."""
    if (features.shape != (330, 160) or labels.shape != (330,) or
            blocks.shape != (330,) or
            not np.array_equal(np.unique(blocks), np.arange(1, 31))):
        raise ValueError("unexpected FaceEMG-11 temporal cache")
    folds = []
    for fit_end, validation_end in ((10, 15), (15, 20)):
        fit = blocks <= fit_end
        validation = (blocks > fit_end) & (blocks <= validation_end)
        folds.append(prepare_fold((features[fit], labels[fit]),
                                  (features[validation], labels[validation]),
                                  features_per_site=8))
    return folds


def run_subject(coarse_cache, temporal_cache, subject, beam_size):
    _, _, _, metadata = load_subject(coarse_cache, subject)
    features, labels, blocks = temporal_subject(
        temporal_cache, subject, metadata["member_sha256"])
    train, validation, query = split_subject(features, labels, blocks)
    random_folds = prepare_cv_folds(*train, features_per_site=8)
    temporal_folds = forward_folds(features, labels, blocks)
    selections = {}
    for name, folds in (("shuffled_cv", random_folds),
                        ("forward_blocks", temporal_folds)):
        started = time.perf_counter()
        result = beam_search(
            OPTIONS,
            lambda programs: score_programs(
                folds, programs, features_per_site=8),
            beam_size=beam_size)
        selections[name] = {
            "channels": result.channels,
            "selection_score": result.score,
            "evaluations": result.evaluations,
            "search_seconds": time.perf_counter() - started,
        }
    fit = (np.vstack((train[0], validation[0])),
           np.r_[train[1], validation[1]])
    query_fold = prepare_fold(fit, query, features_per_site=8)
    programs = np.asarray([selections[name]["channels"]
                           for name in selections], dtype=np.int16)
    query_scores = score_programs(
        [query_fold], programs, features_per_site=8)
    for name, score in zip(selections, query_scores):
        selections[name]["later_block_balanced_accuracy"] = float(score)
    all_channels = score_programs(
        [query_fold], np.arange(20, dtype=np.int16)[None, :],
        features_per_site=8)
    return {
        "source_member_sha256": metadata["member_sha256"],
        "selections": selections,
        "all_twenty_later_block_balanced_accuracy": float(all_channels[0]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse-cache", type=Path, required=True)
    parser.add_argument("--temporal-cache", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", default=[
        "01", "02", "03", "04", "05", "06"])
    parser.add_argument("--beam-size", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    if not set(args.subjects).issubset({f"{i:02d}" for i in range(1, 13)}):
        parser.error("unknown FaceEMG-11 subject")
    result = {
        "schema": "openalterego.faceemg11-forward-search.v1",
        "scope": "Retrospective within-recording facial-command diagnostic; no fresh holdout, silent speech, or reseating",
        "selection": {
            "shuffled_cv": "blocks 1-15, three stratified folds",
            "forward_blocks": "fit blocks 1-10 / validate 11-15; fit 1-15 / validate 16-20",
            "beam_size": args.beam_size,
            "ridge_alpha": 10.0,
        },
        "evaluation": "fit blocks 1-20, inspect exposed blocks 21-30",
        "subjects": {},
    }
    for subject in args.subjects:
        row = run_subject(args.coarse_cache, args.temporal_cache,
                          subject, args.beam_size)
        result["subjects"][subject] = row
        print(subject, {name: round(outcome["later_block_balanced_accuracy"], 4)
                        for name, outcome in row["selections"].items()},
              flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
