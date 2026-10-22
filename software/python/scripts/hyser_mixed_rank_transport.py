"""Diagnose support-to-query ranking over complete mixed Hyser programs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from openalterego.research.dense_programs import prepare_cv_folds, prepare_fold, score_programs
from openalterego.research.supported_programs import prepare_supported_fold
from scripts.hyser_compositional_search import edge_mode_options
from scripts.hyser_mixed_program_pilot import load_mixed
from scripts.hyser_support_conditioned_search import sample_programs
from scripts.hyser_two_support_robust_search import two_support_split


def split_remaining_repetitions(labels: np.ndarray, query_ids: np.ndarray
                                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Alternate post-support repetitions within each class."""
    first, second, excluded = [], [], []
    for label in np.unique(labels[query_ids]):
        indices = query_ids[labels[query_ids] == label]
        if len(indices) < 2:
            excluded.extend(map(int, indices))
            continue
        first.extend(map(int, indices[::2]))
        second.extend(map(int, indices[1::2]))
    if (not first or not second or set(first) & set(second) or
            (set(first) | set(second)) & set(excluded) or
            set(first) | set(second) | set(excluded) != set(map(int, query_ids))):
        raise ValueError("query partitions do not cover disjoint repetitions")
    return np.asarray(first), np.asarray(second), np.asarray(excluded, dtype=int)


def top_overlap(left: np.ndarray, right: np.ndarray, fraction: float = .1) -> float:
    size = max(1, int(np.ceil(len(left) * fraction)))
    a = set(np.argsort(left, kind="stable")[-size:])
    b = set(np.argsort(right, kind="stable")[-size:])
    return len(a & b) / size


def analyze_subject(mono_cache: Path, pair_cache: Path, subject: str,
                    count: int) -> dict:
    day1, meta1 = load_mixed(mono_cache, pair_cache, subject, 1)
    day2, meta2 = load_mixed(mono_cache, pair_cache, subject, 2)
    if (meta1["site_names"] != meta2["site_names"] or
            meta1["pairs"] != meta2["pairs"] or
            meta1["source_record_sha256"] == meta2["source_record_sha256"]):
        raise ValueError("day pair does not align")
    first, second, support_ids, query_ids = two_support_split(day2[1])
    query_a_ids, query_b_ids, excluded_ids = split_remaining_repetitions(
        day2[1], query_ids)
    programs = sample_programs(edge_mode_options(meta1), count, 42)
    cv = score_programs(prepare_cv_folds(*day1, features_per_site=2),
                        programs, features_per_site=2)
    transfer = [score_programs(
        [prepare_fold(day1, (day2[0][ids], day2[1][ids]), features_per_site=2)],
        programs, features_per_site=2) for ids in (first, second)]
    selection = .5 * cv + .25 * (transfer[0] + transfer[1])
    support = (day2[0][support_ids], day2[1][support_ids])

    def query_scores(ids):
        fold = prepare_supported_fold(
            day1, support, (day2[0][ids], day2[1][ids]),
            features_per_site=2, support_weight=4.0)
        return score_programs([fold], programs, features_per_site=2)

    query_a = query_scores(query_a_ids)
    query_b = query_scores(query_b_ids)
    # Both query partitions contain every class, so this is the full-query
    # balanced accuracy weighted by their classwise repetition counts.
    full_query = query_scores(query_ids)
    selected = int(np.argmax(selection))
    oracle_a = int(np.argmax(query_a))
    oracle_full = int(np.argmax(full_query))
    return {
        "day1_source_record_sha256": meta1["source_record_sha256"],
        "day2_source_record_sha256": meta2["source_record_sha256"],
        "candidate_pool_sha256": hashlib.sha256(programs.tobytes()).hexdigest(),
        "support_indices": support_ids.tolist(),
        "query_a_indices": query_a_ids.tolist(),
        "query_b_indices": query_b_ids.tolist(),
        "excluded_from_partition_indices": excluded_ids.tolist(),
        "selection_vs_query_a_spearman": float(spearmanr(selection, query_a).statistic),
        "selection_vs_query_b_spearman": float(spearmanr(selection, query_b).statistic),
        "query_a_vs_query_b_spearman": float(spearmanr(query_a, query_b).statistic),
        "top_tenth_overlap": {
            "selection_query_a": top_overlap(selection, query_a),
            "selection_query_b": top_overlap(selection, query_b),
            "query_a_query_b": top_overlap(query_a, query_b),
        },
        "candidate_full_query_median": float(np.median(full_query)),
        "candidate_full_query_oracle": float(full_query[oracle_full]),
        "selected": {"index": selected, "program": programs[selected].tolist(),
                     "selection_score": float(selection[selected]),
                     "query_a": float(query_a[selected]),
                     "query_b": float(query_b[selected]),
                     "full_query": float(full_query[selected])},
        "query_a_oracle": {
            "index": oracle_a, "query_a": float(query_a[oracle_a]),
            "query_b": float(query_b[oracle_a]),
            "full_query": float(full_query[oracle_a]),
        },
        "query_full_oracle_index": oracle_full,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mono-cache", type=Path, required=True)
    parser.add_argument("--pair-cache", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--candidates", type=int, default=5000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.candidates < 2:
        parser.error("new output and at least two candidates required")
    people = {}
    for subject in args.subjects:
        people[subject] = analyze_subject(args.mono_cache, args.pair_cache,
                                          subject, args.candidates)
        row = people[subject]
        print(subject, round(row["selection_vs_query_b_spearman"], 3),
              round(row["query_a_vs_query_b_spearman"], 3),
              round(row["selected"]["full_query"], 3), flush=True)
    result = {
        "schema": "openalterego.hyser-mixed-rank-transport.v1",
        "scope": "Retrospective exposed forearm mixed-signal programs; query labels used only for diagnostics, never selection",
        "candidate_count": args.candidates, "candidate_seed": 42,
        "support_weight": 4.0,
        "subjects": people,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
