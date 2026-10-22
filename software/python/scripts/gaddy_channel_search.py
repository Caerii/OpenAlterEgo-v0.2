"""Retrospective six-of-eight Gaddy vocalized-energy channel search.

Select on one recording session, write an exclusive choice, then load a later
session. All sessions used here have been exposed in earlier model studies.
"""

import argparse
import hashlib
import itertools
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from openalterego.research.dense_regression import (
    predict_regression_programs, prepare_regression_fold, regression_mae)
from scripts.gaddy_energy_model_compare import session_rows


SIX = np.asarray(list(itertools.combinations(range(8), 6)), dtype=np.int16)
FIXED = (0, 1, 2, 4, 5, 7)
LAGS = (0, 1, 2, 4)


def channel_local_lags(features):
    indices = np.arange(len(features))
    return np.concatenate([
        np.column_stack([features[np.maximum(indices - lag, 0), channel],
                         features[np.maximum(indices - lag, 0), 8 + channel]])
        for channel in range(8) for lag in LAGS], axis=1)


def normalize_rows(rows, calibration_rows=None):
    reference = np.concatenate([row[0] for row in (
        rows if calibration_rows is None else calibration_rows)])
    median = np.median(reference, axis=0)
    q25, q75 = np.quantile(reference, [0.25, 0.75], axis=0)
    scale = np.maximum(q75 - q25, 0.1)
    return [((row[0] - median) / scale, row[1]) for row in rows]


def design(rows):
    return (np.concatenate([channel_local_lags(row[0]) for row in rows]),
            np.concatenate([row[1] for row in rows]))


def manifest_identity(path):
    payload = (path / "manifest.json").read_bytes()
    manifest = json.loads(payload)
    if manifest.get("schema") != "openalterego.gaddy-voiced-subset.v1":
        raise ValueError(f"unrecognized subset manifest: {path}")
    return {"name": path.name,
            "manifest_sha256": hashlib.sha256(payload).hexdigest(),
            "source_md5": manifest["source_md5"],
            "source_path": manifest["source_path"]}


def load(path, cache_dir):
    return session_rows(path, feature_backend="rust", estimate_pitch=False,
                        cache_dir=cache_dir)


def freeze(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--calibration-utterances", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_name(
            f"{args.output.stem}-selection.json").exists():
        parser.error("output or selection record already exists")
    paths = (args.train.resolve(), args.selection.resolve(),
             args.query.resolve())
    if len(set(paths)) != 3 or args.calibration_utterances < 1:
        parser.error("three distinct sessions and positive calibration budget required")
    sources = [manifest_identity(path) for path in paths[:2]]
    if (sources[0]["source_path"] == sources[1]["source_path"] or
            sources[0]["source_md5"] != sources[1]["source_md5"]):
        parser.error("training and selection provenance differs")
    train_rows = load(paths[0], args.cache_dir)
    selection_rows = load(paths[1], args.cache_dir)
    budget = args.calibration_utterances
    if len(selection_rows) <= budget:
        parser.error("selection calibration consumes all utterances")
    fit = design(normalize_rows(train_rows))
    select = design(normalize_rows(
        selection_rows[budget:], selection_rows[:budget]))
    selection_fold = prepare_regression_fold(
        fit, select, features_per_channel=8)
    scores = regression_mae(selection_fold, SIX)
    index = min(range(len(SIX)),
                key=lambda i: (scores[i], tuple(map(int, SIX[i]))))
    chosen = tuple(map(int, SIX[index]))
    selection_record = {
        "schema": "openalterego.gaddy-channel-selection.v1",
        "registered_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Local exclusive choice before query load; sessions exposed in earlier research",
        "train": sources[0], "selection": sources[1],
        "calibration_utterances": budget,
        "ridge_alpha": 100.0,
        "lags": LAGS,
        "programs_scored": len(SIX),
        "selected_channels": chosen,
        "selected_session_mae": float(scores[index]),
    }
    selection_path = args.output.with_name(
        f"{args.output.stem}-selection.json")
    freeze(selection_path, selection_record)

    query_source = manifest_identity(paths[2])
    if (query_source["source_md5"] != sources[0]["source_md5"] or
            query_source["source_path"] in {source["source_path"]
                                            for source in sources}):
        raise ValueError("query provenance differs")
    query_rows = load(paths[2], args.cache_dir)
    if len(query_rows) <= budget:
        raise ValueError("query calibration consumes all utterances")
    query = design(normalize_rows(query_rows[budget:], query_rows[:budget]))
    query_fold = prepare_regression_fold(fit, query, features_per_channel=8)
    programs = {"selected_six": chosen, "fixed_six": FIXED}
    predictions = {}
    for name, channels in programs.items():
        prediction = predict_regression_programs(
            query_fold, np.asarray([channels], dtype=np.int16))[:, 0]
        predictions[name] = prediction
    predictions["all_eight"] = predict_regression_programs(
        query_fold, np.arange(8, dtype=np.int16)[None, :])[:, 0]
    targets = query[1]
    offsets = np.r_[0, np.cumsum([len(row[1]) for row in
                                    query_rows[budget:]])]
    outcomes = {}
    for name, prediction in predictions.items():
        outcomes[name] = {
            "frame_mae": float(np.mean(np.abs(targets - prediction))),
            "utterance_mae": [float(np.mean(np.abs(
                targets[offsets[i]:offsets[i + 1]] -
                prediction[offsets[i]:offsets[i + 1]])))
                for i in range(len(offsets) - 1)],
        }
    report = {"schema": "openalterego.gaddy-channel-search.v1",
              "scope": "Retrospective vocalized acoustic energy, one speaker; virtual subsets of recorded channels, no physical contact map or silent prosody",
              "selection_record": str(selection_path),
              "selection_record_sha256": hashlib.sha256(
                  selection_path.read_bytes()).hexdigest(),
              "query": query_source,
              "query_calibration_utterances": budget,
              "query_evaluation_utterances": len(offsets) - 1,
              "outcomes": outcomes}
    freeze(args.output, report)
    print(json.dumps({name: row["frame_mae"] for name, row in
                      outcomes.items()}, indent=2))


if __name__ == "__main__":
    main()
