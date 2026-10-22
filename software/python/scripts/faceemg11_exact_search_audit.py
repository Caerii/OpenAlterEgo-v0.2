"""Audit beam-search selection regret against all six-of-twenty programs.

Retrospective FaceEMG-11 command diagnostic. Exhaustive selection scores are
never described as a held-out performance estimate or hardware intervention.
"""

import argparse
import itertools
import json
import platform
import time
from pathlib import Path

import numpy as np

from openalterego.research.compositional_search import beam_search
from openalterego.research.dense_programs import prepare_fold, score_programs
from scripts.faceemg11_forward_search import OPTIONS, forward_folds
from scripts.faceemg11_multifidelity_search import temporal_subject
from scripts.faceemg11_program_search import load_subject, split_subject


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse-cache", type=Path, required=True)
    parser.add_argument("--temporal-cache", type=Path, required=True)
    parser.add_argument("--subject", default="01")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.chunk_size < 1:
        parser.error("new output and positive chunk size required")
    _, _, _, metadata = load_subject(args.coarse_cache, args.subject)
    features, labels, blocks = temporal_subject(
        args.temporal_cache, args.subject, metadata["member_sha256"])
    folds = forward_folds(features, labels, blocks)
    started = time.perf_counter()
    beam = beam_search(OPTIONS, lambda programs: score_programs(
        folds, programs, features_per_site=8), beam_size=4)
    beam_seconds = time.perf_counter() - started
    programs = np.asarray(list(itertools.combinations(range(20), 6)),
                          dtype=np.int16)
    scores = np.empty(len(programs), dtype=np.float64)
    started = time.perf_counter()
    for start in range(0, len(programs), args.chunk_size):
        stop = min(start + args.chunk_size, len(programs))
        scores[start:stop] = score_programs(
            folds, programs[start:stop], features_per_site=8)
        if stop % 4096 == 0 or stop == len(programs):
            print(f"scored {stop}/{len(programs)}", flush=True)
    exact_seconds = time.perf_counter() - started
    beam_index = next(i for i, row in enumerate(programs)
                      if tuple(row) == tuple(beam.channels))
    winner = int(np.argmax(scores))
    top_ties = np.flatnonzero(np.isclose(
        scores, scores[winner], atol=1e-12, rtol=0))
    if not np.isclose(scores[beam_index], beam.score, atol=1e-12):
        raise ValueError("beam score disagrees with exhaustive scorer")
    train, validation, query = split_subject(features, labels, blocks)
    fit = (np.vstack((train[0], validation[0])),
           np.r_[train[1], validation[1]])
    query_fold = prepare_fold(fit, query, features_per_site=8)
    query_programs = np.vstack((programs[beam_index], programs[winner]))
    query_scores = score_programs([query_fold], query_programs,
                                  features_per_site=8)
    all_twenty = score_programs(
        [query_fold], np.arange(20, dtype=np.int16)[None, :],
        features_per_site=8)[0]
    tied_query_scores = score_programs(
        [query_fold], programs[top_ties], features_per_site=8)
    result = {
        "schema": "openalterego.faceemg11-exact-search-audit.v1",
        "scope": "Retrospective within-recording selection-score audit; 20 facial channels, command labels, no silent speech or physical intervention",
        "source_member_sha256": metadata["member_sha256"],
        "subject": args.subject,
        "selection": "forward folds fit blocks 1-10 / validate 11-15; fit 1-15 / validate 16-20",
        "channels": 20, "program_size": 6,
        "programs_exhaustively_scored": len(programs),
        "features_per_channel": 8, "ridge_alpha": 10.0,
        "beam_width": 4, "beam_program_evaluations": beam.evaluations,
        "beam_channels": beam.channels, "beam_selection_score": float(beam.score),
        "beam_exact_rank": int(1 + np.sum(scores > scores[beam_index])),
        "exact_best_channels": tuple(map(int, programs[winner])),
        "exact_best_selection_score": float(scores[winner]),
        "exact_top_score_tie_count": len(top_ties),
        "later_block_accuracy_top_tie_min": float(tied_query_scores.min()),
        "later_block_accuracy_top_tie_max": float(tied_query_scores.max()),
        "selection_score_regret": float(scores[winner] - beam.score),
        "later_block_accuracy_beam": float(query_scores[0]),
        "later_block_accuracy_exact_selection": float(query_scores[1]),
        "later_block_accuracy_all_twenty": float(all_twenty),
        "score_quantiles": [float(value) for value in np.quantile(
            scores, [0, 0.25, 0.5, 0.75, 0.95, 0.99, 1])],
        "beam_seconds": beam_seconds, "exhaustive_seconds": exact_seconds,
        "machine": platform.platform(), "numpy": np.__version__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
