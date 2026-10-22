"""Retrospective rolling-origin channel selection over Gaddy sessions."""

import argparse
import json
from pathlib import Path

import numpy as np

from openalterego.research.dense_regression import (
    prepare_regression_fold, regression_mae)
from scripts.gaddy_channel_search import (
    FIXED, SIX, design, load, manifest_identity, normalize_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--ordered-sessions", type=Path, nargs="+", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--calibration-utterances", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or len(args.ordered_sessions) < 2:
        parser.error("new output and at least two ordered sessions required")
    all_paths = [args.train.resolve(), *(
        path.resolve() for path in args.ordered_sessions)]
    if len(set(all_paths)) != len(all_paths):
        parser.error("all session paths must differ")
    fit = design(normalize_rows(load(all_paths[0], args.cache_dir)))
    past_scores = []
    past_sources = []
    report = {"schema": "openalterego.gaddy-channel-rolling.v1",
              "scope": "Retrospective vocalized speech; all sessions previously exposed; program choice uses only earlier sessions within this replay",
              "train": manifest_identity(all_paths[0]),
              "calibration_utterances": args.calibration_utterances,
              "sessions": []}
    fixed_index = next(i for i, program in enumerate(SIX)
                       if tuple(program) == FIXED)
    for path in all_paths[1:]:
        source = manifest_identity(path)
        if source["source_md5"] != report["train"]["source_md5"]:
            parser.error("source archive differs")
        choices = {}
        if past_scores:
            previous = np.asarray(past_scores)
            for name, objective in (
                    ("earliest", previous[0]),
                    ("mean", previous.mean(axis=0)),
                    ("worst", previous.max(axis=0))):
                index = min(range(len(SIX)), key=lambda i: (
                    objective[i], tuple(map(int, SIX[i]))))
                choices[name] = index
        rows = load(path, args.cache_dir)
        if len(rows) <= args.calibration_utterances:
            parser.error("calibration consumes session")
        evaluation = design(normalize_rows(
            rows[args.calibration_utterances:],
            rows[:args.calibration_utterances]))
        fold = prepare_regression_fold(fit, evaluation,
                                       features_per_channel=8)
        scores = regression_mae(fold, SIX)
        all_eight = regression_mae(
            fold, np.arange(8, dtype=np.int16)[None, :])[0]
        entry = {
            "source": source,
            "earlier_selection_sessions": [s["name"] for s in past_sources],
            "evaluation_utterances": len(rows) - args.calibration_utterances,
            "choices": {
                name: {"channels": tuple(map(int, SIX[index])),
                       "query_mae": float(scores[index]),
                       "query_rank_of_28": int(np.argsort(
                           np.argsort(scores))[index] + 1)}
                for name, index in choices.items()},
            "fixed_six_mae": float(scores[fixed_index]),
            "all_eight_mae": float(all_eight),
            "query_oracle_mae": float(scores.min()),
        }
        report["sessions"].append(entry)
        past_sources.append(source)
        past_scores.append(scores)
        print(path.name, entry["choices"], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
