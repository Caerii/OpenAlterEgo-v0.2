"""Replay one frozen six-channel choice on additional exposed Gaddy sessions."""

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
    parser.add_argument("--queries", type=Path, nargs="+", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--selection-record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    if len({path.resolve() for path in [args.train, *args.queries]}) != (
            len(args.queries) + 1):
        parser.error("training and query sessions must differ")
    selection = json.loads(args.selection_record.read_text(encoding="utf-8"))
    source = manifest_identity(args.train)
    if (selection.get("schema") != "openalterego.gaddy-channel-selection.v1" or
            source != selection["train"]):
        parser.error("frozen choice provenance differs")
    selected = tuple(selection["selected_channels"])
    selected_index = next(i for i, program in enumerate(SIX)
                          if tuple(program) == selected)
    fixed_index = next(i for i, program in enumerate(SIX)
                       if tuple(program) == FIXED)
    fit = design(normalize_rows(load(args.train, args.cache_dir)))
    budget = selection["calibration_utterances"]
    report = {"schema": "openalterego.gaddy-channel-frozen-replay.v1",
              "scope": "Additional exposed vocalized sessions; channel choice fixed by earlier selection session",
              "selection_record": str(args.selection_record),
              "selected_channels": selected, "fixed_channels": FIXED,
              "calibration_utterances": budget, "sessions": {}}
    for path in args.queries:
        identity = manifest_identity(path)
        if (identity["source_md5"] != source["source_md5"] or
                identity["source_path"] in (
                    source["source_path"],
                    selection["selection"]["source_path"])):
            parser.error("query provenance differs")
        rows = load(path, args.cache_dir)
        if len(rows) <= budget:
            parser.error("calibration budget consumes query session")
        query = design(normalize_rows(rows[budget:], rows[:budget]))
        fold = prepare_regression_fold(fit, query, features_per_channel=8)
        scores = regression_mae(fold, SIX)
        all_eight = regression_mae(
            fold, np.arange(8, dtype=np.int16)[None, :])[0]
        rank = int(np.argsort(np.argsort(scores))[selected_index] + 1)
        oracle_index = int(np.argmin(scores))
        report["sessions"][path.name] = {
            "source": identity,
            "evaluation_utterances": len(rows) - budget,
            "selected_six_mae": float(scores[selected_index]),
            "fixed_six_mae": float(scores[fixed_index]),
            "all_eight_mae": float(all_eight),
            "selected_rank_of_28": rank,
            "query_oracle_channels": tuple(map(int, SIX[oracle_index])),
            "query_oracle_mae": float(scores[oracle_index]),
            "selected_minus_oracle_mae": float(
                scores[selected_index] - scores[oracle_index]),
        }
        print(path.name, report["sessions"][path.name], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
