"""Retrospective multi-session selection of six Gaddy EMG channels."""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from openalterego.research.dense_regression import (
    prepare_regression_fold, regression_mae)
from scripts.gaddy_channel_search import (
    FIXED, SIX, design, load, manifest_identity, normalize_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--selection-sessions", type=Path, nargs="+", required=True)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--calibration-utterances", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selection_path = args.output.with_name(
        f"{args.output.stem}-selection.json")
    if args.output.exists() or selection_path.exists():
        parser.error("output or selection record already exists")
    paths = [args.train.resolve(), *(
        path.resolve() for path in args.selection_sessions),
        args.query.resolve()]
    if (len(set(paths)) != len(paths) or
            len(args.selection_sessions) < 2 or
            args.calibration_utterances < 1):
        parser.error("distinct train, two or more selection, and query sessions required")
    sources = [manifest_identity(path) for path in paths[:-1]]
    if (len({source["source_path"] for source in sources}) != len(sources) or
            len({source["source_md5"] for source in sources}) != 1):
        parser.error("selection provenance differs")
    fit = design(normalize_rows(load(paths[0], args.cache_dir)))
    score_rows = []
    for path in paths[1:-1]:
        rows = load(path, args.cache_dir)
        if len(rows) <= args.calibration_utterances:
            parser.error("calibration consumes a selection session")
        validation = design(normalize_rows(
            rows[args.calibration_utterances:],
            rows[:args.calibration_utterances]))
        fold = prepare_regression_fold(fit, validation,
                                       features_per_channel=8)
        score_rows.append(regression_mae(fold, SIX))
    scores = np.asarray(score_rows)
    objectives = {"mean": scores.mean(axis=0), "worst": scores.max(axis=0)}
    choices = {}
    for name, objective in objectives.items():
        index = min(range(len(SIX)), key=lambda i: (
            objective[i], tuple(map(int, SIX[i]))))
        choices[name] = tuple(map(int, SIX[index]))
    selection_record = {
        "schema": "openalterego.gaddy-multisession-channel-selection.v1",
        "registered_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Local exclusive record; every session was exposed by earlier research",
        "train": sources[0], "selection_sessions": sources[1:],
        "calibration_utterances": args.calibration_utterances,
        "ridge_alpha": 100.0,
        "objectives": ["mean_session_mae", "worst_session_mae"],
        "choices": choices,
        "selection_mae_by_session": {
            source["name"]: [float(value) for value in row]
            for source, row in zip(sources[1:], scores)},
    }
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    with selection_path.open("x", encoding="utf-8") as stream:
        json.dump(selection_record, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())

    query_source = manifest_identity(paths[-1])
    if (query_source["source_md5"] != sources[0]["source_md5"] or
            query_source["source_path"] in {
                source["source_path"] for source in sources}):
        raise ValueError("query provenance differs")
    rows = load(paths[-1], args.cache_dir)
    if len(rows) <= args.calibration_utterances:
        raise ValueError("calibration consumes query session")
    query = design(normalize_rows(
        rows[args.calibration_utterances:], rows[:args.calibration_utterances]))
    fold = prepare_regression_fold(fit, query, features_per_channel=8)
    query_scores = regression_mae(fold, SIX)
    all_eight = regression_mae(
        fold, np.arange(8, dtype=np.int16)[None, :])[0]
    selected = {}
    for name, channels in (choices | {"fixed": FIXED}).items():
        index = next(i for i, program in enumerate(SIX)
                     if tuple(program) == channels)
        selected[name] = {
            "channels": channels,
            "query_mae": float(query_scores[index]),
            "query_rank_of_28": int(np.argsort(
                np.argsort(query_scores))[index] + 1),
        }
    oracle = int(np.argmin(query_scores))
    report = {
        "schema": "openalterego.gaddy-multisession-channel-search.v1",
        "scope": "Post hoc vocalized-energy comparison; no fresh holdout or physical intervention",
        "selection_record": str(selection_path),
        "query": query_source,
        "query_evaluation_utterances": len(rows) - args.calibration_utterances,
        "selected": selected,
        "all_eight_query_mae": float(all_eight),
        "query_oracle_channels": tuple(map(int, SIX[oracle])),
        "query_oracle_mae": float(query_scores[oracle]),
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report["selected"] | {
        "all_eight_query_mae": report["all_eight_query_mae"]}, indent=2))


if __name__ == "__main__":
    main()
