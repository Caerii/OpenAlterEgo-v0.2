"""Retrospective leave-one-subject-out search-prior diagnostic."""

import argparse
import json
from pathlib import Path

import numpy as np

from openalterego.research.cross_subject_prior import leave_one_out_prior
from openalterego.research.program_search import normalize_batches, ridge_proxy_rank
from scripts.silentwear_program_search import subject_sessions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", default=["S01", "S02", "S03"])
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    if sorted(args.subjects) != ["S01", "S02", "S03"]:
        parser.error("this development diagnostic requires exactly S01–S03")
    personal, future, manifests = {}, {}, {}
    keys = None
    for subject in args.subjects:
        load_session, provenance, manifest_hash = subject_sessions(
            args.root, subject, args.cache_dir)
        session = {day: load_session(day) for day in (1, 2, 3)}
        forward, backward, next_day = {}, {}, {}
        for norm in ("none", "recording_robust"):
            first = normalize_batches(session[1], norm)
            second = normalize_batches(session[2], norm)
            third = normalize_batches(session[3], norm)
            forward.update({(norm, program.channels): score
                            for program, score in ridge_proxy_rank(first, second,
                                                                   norm)})
            backward.update({(norm, program.channels): score
                             for program, score in ridge_proxy_rank(second, first,
                                                                    norm)})
            joined = normalize_batches(session[1] + session[2], norm)
            next_day.update({(norm, program.channels): score
                             for program, score in ridge_proxy_rank(joined, third,
                                                                    norm)})
        current_keys = sorted(forward)
        if keys is None:
            keys = current_keys
        elif keys != current_keys:
            raise ValueError("subject program spaces differ")
        personal[subject] = np.asarray([min(forward[key], backward[key])
                                        for key in keys])
        future[subject] = np.asarray([next_day[key] for key in keys])
        manifests[subject] = {"input_manifest_sha256": manifest_hash,
                              "inputs": provenance}
        print(subject, "scored", len(keys), "structures", flush=True)
    results = leave_one_out_prior(keys, personal, future)
    for subject, row in results.items():
        print(subject, row["donor_target_spearman"],
              {name: method["target_future_proxy_score"] for name, method in
               row["methods"].items()}, flush=True)
    result = {
        "schema": "openalterego.cross-subject-prior.v1",
        "scope": "Retrospective development comparison; recorded-channel ridge proxy scores, not logistic outcomes or physical counterfactuals.",
        "subjects": {subject: {**results[subject], **manifests[subject]}
                     for subject in args.subjects},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
