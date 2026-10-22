"""Post hoc rank-transport audit of a frozen Gaddy six-channel choice."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from openalterego.research.dense_regression import (
    prepare_regression_fold, regression_mae)
from scripts.gaddy_channel_search import (
    FIXED, SIX, design, load, manifest_identity, normalize_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--selection-record", type=Path, required=True)
    parser.add_argument("--outcome-record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    selection = json.loads(args.selection_record.read_text(encoding="utf-8"))
    outcome = json.loads(args.outcome_record.read_text(encoding="utf-8"))
    paths = (args.train.resolve(), args.selection.resolve(), args.query.resolve())
    sources = [manifest_identity(path) for path in paths]
    if (selection.get("schema") != "openalterego.gaddy-channel-selection.v1" or
            outcome.get("schema") != "openalterego.gaddy-channel-search.v1" or
            sources != [selection["train"], selection["selection"],
                        outcome["query"]] or
            len(set(paths)) != 3):
        parser.error("frozen selection and source provenance differ")
    budget = selection["calibration_utterances"]
    if budget != outcome["query_calibration_utterances"]:
        parser.error("selection and query calibration budgets differ")
    rows = [load(path, args.cache_dir) for path in paths]
    fit = design(normalize_rows(rows[0]))
    score_rows = [design(normalize_rows(group[budget:], group[:budget]))
                  for group in rows[1:]]
    scores = [regression_mae(
        prepare_regression_fold(fit, row, features_per_channel=8), SIX)
        for row in score_rows]
    selected = tuple(selection["selected_channels"])
    selected_index = next(i for i, program in enumerate(SIX)
                          if tuple(program) == selected)
    best_query_index = int(np.argmin(scores[1]))
    ranks = np.argsort(np.argsort(scores[1])) + 1
    errors = outcome["outcomes"]
    selected_error = np.asarray(errors["selected_six"]["utterance_mae"])
    rng = np.random.default_rng(42)
    bootstrap = rng.integers(0, len(selected_error),
                             size=(5000, len(selected_error)))
    paired = {}
    for name in ("fixed_six", "all_eight"):
        difference = np.asarray(errors[name]["utterance_mae"]) - selected_error
        paired[name] = {
            "control_minus_selected_mean_utterance_mae": float(difference.mean()),
            "bootstrap_ci95": list(map(float, np.quantile(
                difference[bootstrap].mean(axis=1), [0.025, 0.975]))),
        }
    report = {
        "schema": "openalterego.gaddy-channel-rank-diagnostic.v1",
        "scope": "Post hoc ranking of all 28 six-channel subsets; query session exposed in previous research",
        "selected_channels": selected,
        "selected_validation_mae": float(scores[0][selected_index]),
        "selected_query_mae": float(scores[1][selected_index]),
        "selected_query_rank_of_28": int(ranks[selected_index]),
        "query_oracle_channels": tuple(map(int, SIX[best_query_index])),
        "query_oracle_mae": float(scores[1][best_query_index]),
        "selected_minus_query_oracle_mae": float(
            scores[1][selected_index] - scores[1][best_query_index]),
        "selection_query_rank_spearman": float(spearmanr(
            scores[0], scores[1]).statistic),
        "fixed_channels": FIXED,
        "paired_utterance_advantage": paired,
        "programs": [{"channels": tuple(map(int, program)),
                      "selection_mae": float(validation),
                      "query_mae": float(query)}
                     for program, validation, query in zip(
                         SIX, scores[0], scores[1])],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in (
        "selected_query_rank_of_28", "query_oracle_mae",
        "selection_query_rank_spearman", "paired_utterance_advantage")},
        indent=2))


if __name__ == "__main__":
    main()
