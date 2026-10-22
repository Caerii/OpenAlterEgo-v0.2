"""Check every prepared-fold candidate score before benchmarking Rust speed."""

import argparse
import hashlib
import importlib
import json
import time
from pathlib import Path

import numpy as np

from openalterego.research.dense_programs import (
    prepare_cv_folds, prepare_fold, score_programs_by_fold)
from openalterego.research.uniform_program_pool import uniform_program_prefix
from scripts.hyser_compositional_search import edge_mode_options
from scripts.hyser_mixed_program_pilot import load_mixed
from scripts.hyser_two_support_robust_search import two_support_split


def benchmark_subject(mono_cache: Path, pair_cache: Path, subject: str,
                      count: int) -> dict:
    native = importlib.import_module("openalterego_accel.openalterego_accel")

    day1, meta1 = load_mixed(mono_cache, pair_cache, subject, 1)
    day2, meta2 = load_mixed(mono_cache, pair_cache, subject, 2)
    if (meta1["site_names"] != meta2["site_names"] or
            meta1["pairs"] != meta2["pairs"] or
            meta1["source_record_sha256"] == meta2["source_record_sha256"]):
        raise ValueError("day pair does not align")
    first, second, _, _ = two_support_split(day2[1])
    programs = uniform_program_prefix(edge_mode_options(meta1), count, 42)
    folds = prepare_cv_folds(*day1, features_per_site=2)
    folds += [prepare_fold(day1, (day2[0][ids], day2[1][ids]),
                           features_per_site=2) for ids in (first, second)]
    timings = {}
    scores = {}
    for backend in ("numpy", "rust"):
        start = time.perf_counter()
        scores[backend] = score_programs_by_fold(
            folds, programs, features_per_site=2, backend=backend)
        timings[backend] = time.perf_counter() - start
    difference = np.abs(scores["numpy"] - scores["rust"])
    selection = {backend: (.5 * scores[backend][:3].mean(axis=0) +
                           .25 * (scores[backend][3] + scores[backend][4]))
                 for backend in scores}
    return {
        "day1_source_record_sha256": meta1["source_record_sha256"],
        "day2_source_record_sha256": meta2["source_record_sha256"],
        "extension_binary_sha256": hashlib.sha256(
            Path(native.__file__).read_bytes()).hexdigest(),
        "candidate_count": count, "fold_count": len(folds),
        "max_absolute_score_difference": float(difference.max()),
        "scores_differing_more_than_1e-12": int(np.count_nonzero(difference > 1e-12)),
        "selection_index": {backend: int(np.argmax(value))
                            for backend, value in selection.items()},
        "score_seconds": timings,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mono-cache", type=Path, required=True)
    parser.add_argument("--pair-cache", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--candidates", type=int, default=5000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.candidates < 1:
        parser.error("new output and positive candidate count required")
    people = {}
    for subject in args.subjects:
        people[subject] = benchmark_subject(
            args.mono_cache, args.pair_cache, subject, args.candidates)
        print(subject, people[subject], flush=True)
    result = {"schema": "openalterego.hyser-dense-rust-parity.v1",
              "scope": "All five prepared-fold accuracy scores per candidate, same source data and candidate pool; local single-run timing",
              "subjects": people}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
