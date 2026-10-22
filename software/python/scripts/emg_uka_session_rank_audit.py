"""Post hoc audit of all 20 EMG-UKA three-channel session-008 programs.

The selection and primary evaluation have already been exposed. This is a
diagnostic of ranking and regret, never a new untouched holdout.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from scripts.emg_uka_session_program import (
    CHANNELS, RESULT_SCHEMA, accuracy, check_source, development_data,
    session, sha256)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    check_source(args.root, args.manifest)
    selection_bytes = args.selection.read_bytes()
    evaluation_bytes = args.evaluation.read_bytes()
    selected = json.loads(selection_bytes)
    evaluation = json.loads(evaluation_bytes)
    if (selected.get("schema") != RESULT_SCHEMA or
            selected.get("phase") != "selection" or
            evaluation.get("phase") != "evaluation" or
            evaluation.get("selection_record_sha256") !=
            sha256(selection_bytes)):
        raise ValueError("rank audit does not match frozen selection/evaluation")
    xs, ys, labels, _ = development_data(args.root)
    query_x, prompts, digest = session(args.root, "008")
    if digest != evaluation["holdout_content_digest"]:
        raise ValueError("holdout content differs from frozen evaluation")
    mapping = {prompt: index for index, prompt in enumerate(labels)}
    query_y = np.asarray([mapping[prompt] for prompt in prompts])
    train_x, train_y = np.vstack(xs), np.concatenate(ys)
    held = np.asarray([accuracy(train_x, train_y, query_x, query_y,
                                program) for program in CHANNELS])
    development = np.asarray([
        np.mean(selected["selection_fold_accuracy"][
            "-".join(map(str, program))]) for program in CHANNELS])
    selected_program = tuple(selected["selected"])
    selected_index = CHANNELS.index(selected_program)
    if held[selected_index] != evaluation["results"]["selected_three"]["accuracy"]:
        raise ValueError("selected program score differs from frozen evaluation")
    top = int(np.argmax(held))
    result = {
        "schema": "openalterego.emg-uka-session-rank-audit.v1",
        "scope": "Post hoc, exposed session 008; 20 virtual recorded-channel programs on ten repeated audible prompts, not physical placement validation",
        "selection_sha256": sha256(selection_bytes),
        "evaluation_sha256": sha256(evaluation_bytes),
        "holdout_content_digest": digest,
        "speaker": "008", "session": "008", "program_size": 3,
        "all_programs": [{"channels": program,
                          "development_mean": float(development[index]),
                          "session008_accuracy": float(held[index])}
                         for index, program in enumerate(CHANNELS)],
        "spearman_development_vs_session008": float(
            spearmanr(development, held).statistic),
        "selected": selected_program,
        "selected_session008_accuracy": float(held[selected_index]),
        "selected_session008_rank": int(1 + np.sum(held > held[selected_index])),
        "posthoc_best": CHANNELS[top],
        "posthoc_best_accuracy": float(held[top]),
        "selected_regret": float(held[top] - held[selected_index]),
        "mean_program_accuracy": float(held.mean()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in (
        "spearman_development_vs_session008", "selected_session008_rank",
        "posthoc_best", "posthoc_best_accuracy", "selected_regret")},
        indent=2))


if __name__ == "__main__":
    main()
