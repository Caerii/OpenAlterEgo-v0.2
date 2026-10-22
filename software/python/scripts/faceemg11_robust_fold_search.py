"""Compare mean and worst-forward-fold sparse facial command selection.

The objective is fixed before later-block scoring, but all recordings were
previously exposed. This is a retrospective method diagnostic only.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from openalterego.research.compositional_search import beam_search
from openalterego.research.dense_programs import (
    prepare_fold, score_programs, score_programs_by_fold)
from scripts.faceemg11_forward_search import OPTIONS, forward_folds
from scripts.faceemg11_multifidelity_search import temporal_subject
from scripts.faceemg11_program_search import load_subject, split_subject


def run_subject(coarse_cache, temporal_cache, subject):
    _, _, _, metadata = load_subject(coarse_cache, subject)
    features, labels, blocks = temporal_subject(
        temporal_cache, subject, metadata["member_sha256"])
    folds = forward_folds(features, labels, blocks)
    choices = {}
    for objective in ("mean", "worst"):
        started = time.perf_counter()
        def score(programs):
            matrix = score_programs_by_fold(
                folds, programs, features_per_site=8)
            return matrix.mean(axis=0) if objective == "mean" else matrix.min(axis=0)
        selected = beam_search(OPTIONS, score, beam_size=4)
        per_fold = score_programs_by_fold(
            folds, np.asarray([selected.channels], dtype=np.int16),
            features_per_site=8)[:, 0]
        choices[objective] = {
            "channels": selected.channels,
            "selection_score": selected.score,
            "forward_fold_scores": [float(value) for value in per_fold],
            "program_evaluations": selected.evaluations,
            "search_seconds": time.perf_counter() - started,
        }
    train, validation, query = split_subject(features, labels, blocks)
    fit = (np.vstack((train[0], validation[0])),
           np.r_[train[1], validation[1]])
    query_fold = prepare_fold(fit, query, features_per_site=8)
    programs = np.asarray([choices[name]["channels"]
                           for name in choices], dtype=np.int16)
    later = score_programs([query_fold], programs, features_per_site=8)
    for name, accuracy in zip(choices, later):
        choices[name]["later_block_balanced_accuracy"] = float(accuracy)
    all_twenty = score_programs(
        [query_fold], np.arange(20, dtype=np.int16)[None, :],
        features_per_site=8)[0]
    return {"source_member_sha256": metadata["member_sha256"],
            "choices": choices,
            "all_twenty_later_block_balanced_accuracy": float(all_twenty)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse-cache", type=Path, required=True)
    parser.add_argument("--temporal-cache", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", default=[
        "01", "02", "03", "04", "05", "06"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.output.exists() or len(set(args.subjects)) != len(args.subjects) or
            not set(args.subjects).issubset({f"{i:02d}" for i in range(1, 13)})):
        parser.error("new output and distinct known subjects required")
    result = {
        "schema": "openalterego.faceemg11-robust-fold-search.v1",
        "scope": "Retrospective facial command method test; no fresh subject holdout, silent speech, or physical layout intervention",
        "selection": "beam-4 six of 20; mean or minimum balanced accuracy across two forward folds, equal cost",
        "forward_folds": "fit 1-10 validate 11-15; fit 1-15 validate 16-20",
        "later_blocks": "fit 1-20, evaluate already exposed 21-30",
        "subjects": {},
    }
    for subject in args.subjects:
        row = run_subject(args.coarse_cache, args.temporal_cache, subject)
        result["subjects"][subject] = row
        print(subject, {name: choice["later_block_balanced_accuracy"]
                        for name, choice in row["choices"].items()}, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
