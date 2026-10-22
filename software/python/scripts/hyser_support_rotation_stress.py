"""Stress test calibration-conditioned program choices across support trials."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from openalterego.research.dense_programs import (
    prepare_cv_folds, prepare_fold, score_programs)
from openalterego.research.hyser_features import site_coordinates
from openalterego.research.supported_programs import prepare_supported_fold
from scripts.hyser_compositional_search import edge_mode_options
from scripts.hyser_mixed_program_pilot import load_mixed, physical_contacts
from scripts.hyser_program_search import fixed_grid_design
from scripts.hyser_support_conditioned_search import sample_programs


def support_at_rank(labels: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """Choose the same within-class repetition rank for each present class."""
    if labels.ndim != 1 or rank < 0:
        raise ValueError("invalid labels or rank")
    support = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        if rank >= len(indices) or len(indices) < 2:
            raise ValueError("class lacks support or query at requested rank")
        support.append(int(indices[rank]))
    support = np.asarray(sorted(support), dtype=np.int64)
    query = np.setdiff1d(np.arange(len(labels)), support, assume_unique=True)
    return support, query


def choose_program(cv: np.ndarray, support_score: np.ndarray,
                   candidates: np.ndarray) -> tuple[int, ...]:
    """Fixed equal-average rule with lexicographic candidate tie break."""
    return tuple(map(int, candidates[np.argmax((cv + support_score) / 2)]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mono-cache", type=Path, required=True)
    parser.add_argument("--pair-cache", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--reference-selection", type=Path,
                        help="Assert rank-zero choices match the frozen method check")
    parser.add_argument("--candidates", type=int, default=5000)
    parser.add_argument("--ranks", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    if args.ranks[0] != 0 or len(set(args.ranks)) != len(args.ranks):
        parser.error("ranks must start with zero and be distinct")
    reference = (json.loads(args.reference_selection.read_text(encoding="utf-8"))
                 if args.reference_selection else None)
    if reference and reference.get("schema") != "openalterego.hyser-support-search-selection.v1":
        parser.error("unexpected reference selection")
    result = {
        "schema": "openalterego.hyser-support-rotation-stress.v1",
        "scope": "Retrospective support sensitivity, not independent validation or chronological calibration",
        "candidate_count": args.candidates, "candidate_seed": 42,
        "support_ranks": args.ranks, "support_weight": 4.0,
        "subjects": {},
    }
    for subject in args.subjects:
        started = time.perf_counter()
        day1, meta1 = load_mixed(args.mono_cache, args.pair_cache, subject, 1)
        day2, meta2 = load_mixed(args.mono_cache, args.pair_cache, subject, 2)
        if (meta1["site_names"] != meta2["site_names"] or
                meta1["pairs"] != meta2["pairs"]):
            raise ValueError(f"session metadata changed: {subject}")
        candidates = sample_programs(edge_mode_options(meta1), args.candidates, 42)
        cv = score_programs(prepare_cv_folds(*day1, features_per_site=2),
                            candidates, features_per_site=2)
        fixed = tuple(fixed_grid_design(
            site_coordinates(tuple(meta1["site_names"]))))
        by_rank = {}
        first_choice = None
        first_contacts = None
        for rank in args.ranks:
            support_ids, query_ids = support_at_rank(day2[1], rank)
            support = (day2[0][support_ids], day2[1][support_ids])
            query = (day2[0][query_ids], day2[1][query_ids])
            transfer = score_programs(
                [prepare_fold(day1, support, features_per_site=2)],
                candidates, features_per_site=2)
            chosen = choose_program(cv, transfer, candidates)
            contacts = physical_contacts(chosen, meta1["pairs"])
            if rank == 0:
                first_choice, first_contacts = chosen, contacts
                if reference:
                    expected = reference["choices"][subject]["programs"][
                        "half_cv_half_support"]
                    if list(chosen) != expected:
                        raise ValueError(f"rank-zero choice differs: {subject}")
            fold = prepare_supported_fold(
                day1, support, query, features_per_site=2, support_weight=4.0)
            designs = np.asarray([chosen, fixed, first_choice, contacts],
                                 dtype=np.int16)
            values = score_programs([fold], designs, features_per_site=2)
            by_rank[str(rank)] = {
                "program": chosen, "array_contacts": contacts,
                "support_trials": len(support_ids), "query_trials": len(query_ids),
                "selected_balanced_accuracy": float(values[0]),
                "fixed_balanced_accuracy": float(values[1]),
                "rank_zero_program_balanced_accuracy": float(values[2]),
                "selected_same_contacts_all_shared_reference": float(values[3]),
                "selected_minus_fixed": float(values[0] - values[1]),
                "exact_program_as_rank_zero": chosen == first_choice,
                "contact_overlap_with_rank_zero": len(set(contacts) &
                                                      set(first_contacts)),
            }
        result["subjects"][subject] = {
            "day1_source_record_sha256": meta1["source_record_sha256"],
            "day2_source_record_sha256": meta2["source_record_sha256"],
            "ranks": by_rank, "elapsed_seconds": time.perf_counter() - started,
        }
        print(subject, [(rank, round(value["selected_minus_fixed"], 3),
                         value["contact_overlap_with_rank_zero"])
                        for rank, value in by_rank.items()], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
