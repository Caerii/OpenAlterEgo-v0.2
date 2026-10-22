"""Matched same-process timing of exact and beam temporal program search."""

import argparse
import json
import time
from pathlib import Path

from openalterego.research.compositional_search import beam_search
from openalterego.research.dense_programs import prepare_cv_folds, score_programs
from scripts.faceemg11_multifidelity_search import temporal_subject
from scripts.faceemg11_program_search import CANDIDATES, load_subject, split_subject
from scripts.faceemg11_stepwise_search import OPTIONS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse-cache", type=Path, required=True)
    parser.add_argument("--temporal-cache", type=Path, required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 2:
        parser.error("output must be new and repeats >= 2")
    _, _, _, metadata = load_subject(args.coarse_cache, args.subject)
    features, labels, groups = temporal_subject(
        args.temporal_cache, args.subject, metadata["member_sha256"])
    train, _, _ = split_subject(features, labels, groups)
    folds = prepare_cv_folds(*train, features_per_site=8)

    def run(method):
        started = time.perf_counter()
        if method == "exact":
            scores = score_programs(folds, CANDIDATES, features_per_site=8)
            score = float(scores.max())
            evaluations = len(CANDIDATES)
        else:
            width = int(method.split("_")[1])
            chosen = beam_search(
                OPTIONS, lambda batch: score_programs(
                    folds, batch, features_per_site=8),
                beam_size=width)
            score = chosen.score
            evaluations = chosen.evaluations
        return {"seconds": time.perf_counter() - started,
                "selected_cv": score, "evaluations": evaluations}

    methods = ("exact", "beam_1", "beam_4", "beam_16")
    for method in methods:
        run(method)
    records = {method: [] for method in methods}
    for repetition in range(args.repeats):
        for method in methods[repetition % len(methods):] + methods[:repetition % len(methods)]:
            value = run(method)
            records[method].append(value)
            print(repetition, method, round(value["seconds"], 3), flush=True)
    result = {
        "schema": "openalterego.faceemg11-search-timing.v1",
        "scope": "Same-process local timing; data load and fold preparation excluded; not a cross-machine benchmark",
        "subject": args.subject,
        "source_member_sha256": metadata["member_sha256"],
        "repeats": args.repeats,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
