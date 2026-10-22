"""Measure whether virtual-montage proxy rankings transfer across sessions."""

import argparse
import json
from pathlib import Path

from openalterego.research.program_search import normalize_batches, ridge_proxy_rank_many
from openalterego.research.rank_stability import batch_robust_choices, compare_rankings
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
    if not set(args.subjects).issubset({"S01", "S02", "S03"}):
        parser.error("S04 remains sealed for this development diagnostic")
    result = {
        "schema": "openalterego.virtual-program-rank-stability.v2",
        "split": "fit session 1; compare proxy rankings on sessions 2 and 3",
        "scope": "Retrospective development diagnostic on existing channels; session 3 is already explored, not a fresh test.",
        "subjects": {},
    }
    for subject in args.subjects:
        load_session, provenance, manifest_hash = subject_sessions(
            args.root, subject, args.cache_dir)
        batches = {session: load_session(session) for session in (1, 2, 3)}
        selection, later = {}, {}
        batch_scores = [{} for _ in batches[2]]
        for norm in ("none", "recording_robust"):
            train = normalize_batches(batches[1], norm)
            validations = [normalize_batches(batches[2], norm)] + [
                normalize_batches([batch], norm) for batch in batches[2]] + [
                normalize_batches(batches[3], norm)]
            for scores, ranking in zip([selection, *batch_scores, later],
                                       ridge_proxy_rank_many(train, validations,
                                                             norm)):
                scores.update({(norm, program.channels): score
                               for program, score in ranking})
        result["subjects"][subject] = {
            "input_manifest_sha256": manifest_hash,
            "comparison": compare_rankings(selection, later),
            "batch_robust_choices": batch_robust_choices(batch_scores, later),
            "inputs": provenance,
        }
        print(subject, result["subjects"][subject]["comparison"], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
