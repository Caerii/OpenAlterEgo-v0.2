"""Retrospective nested-session comparison of three-channel selection rules.

All eight speaker-008 sessions are now exposed. Each outer target is excluded
from its own inner program selection and classifier fit, but this comparison
is development analysis rather than a new independent validation.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.emg_uka_session_program import (
    CHANNELS, FIXED, accuracy, check_source, classify, session)


SESSIONS = tuple(f"{index:03d}" for index in range(1, 9))
RULES = ("mean", "mean_minus_std", "worst_two_mean", "minimum")


def rank_score(fold_scores: np.ndarray, rule: str) -> np.ndarray:
    if fold_scores.shape[1] < 2:
        raise ValueError("need multiple independent inner sessions")
    if rule == "mean":
        return fold_scores.mean(axis=1)
    if rule == "mean_minus_std":
        return fold_scores.mean(axis=1) - fold_scores.std(axis=1)
    if rule == "worst_two_mean":
        return np.sort(fold_scores, axis=1)[:, :2].mean(axis=1)
    if rule == "minimum":
        return fold_scores.min(axis=1)
    raise ValueError("unknown program selection rule")


def choose(fold_scores: np.ndarray, rule: str) -> int:
    scores = rank_score(fold_scores, rule)
    means = fold_scores.mean(axis=1)
    minima = fold_scores.min(axis=1)
    return min(range(len(CHANNELS)),
               key=lambda i: (-scores[i], -means[i] if rule != "mean" else
                              -minima[i], CHANNELS[i]))


def nested_target(xs, ys, target: int) -> dict:
    other = [index for index in range(len(SESSIONS)) if index != target]
    inner = np.empty((len(CHANNELS), len(other)), dtype=np.float64)
    for fold, validation in enumerate(other):
        train = [index for index in other if index != validation]
        train_x = np.vstack([xs[index] for index in train])
        train_y = np.concatenate([ys[index] for index in train])
        for program_index, program in enumerate(CHANNELS):
            inner[program_index, fold] = accuracy(
                train_x, train_y, xs[validation], ys[validation], program)
    train_x = np.vstack([xs[index] for index in other])
    train_y = np.concatenate([ys[index] for index in other])
    held_correct = np.asarray([
        np.sum(classify(train_x, train_y, xs[target], program) == ys[target])
        for program in CHANNELS], dtype=np.int16)
    selected = {rule: CHANNELS[choose(inner, rule)] for rule in RULES}
    fixed = FIXED
    random = CHANNELS[int(np.random.default_rng(42).integers(len(CHANNELS)))]
    selections = {**selected, "fixed": fixed, "random_seed42": random}
    scores = {rule: int(held_correct[CHANNELS.index(program)])
              for rule, program in selections.items()}
    scores["all_six"] = int(np.sum(classify(
        train_x, train_y, xs[target], tuple(range(6))) == ys[target]))
    return {
        "target_session": SESSIONS[target],
        "inner_session_order": [SESSIONS[index] for index in other],
        "inner_correct_out_of_ten": np.rint(inner * 10).astype(int).tolist(),
        "selected_programs": selections,
        "outer_correct_out_of_ten": scores,
        "posthoc_oracle_correct_out_of_ten": int(held_correct.max()),
        "posthoc_oracle_programs": [CHANNELS[index] for index in
                                    np.flatnonzero(held_correct ==
                                                    held_correct.max())],
        "all_programs_outer_correct_out_of_ten": held_correct.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    manifest = check_source(args.root, args.manifest)
    loaded = [session(args.root, session_id) for session_id in SESSIONS]
    labels = tuple(sorted(loaded[0][1]))
    if any(set(prompts) != set(labels) for _, prompts, _ in loaded):
        raise ValueError("sessions do not have the same ten prompts")
    label_index = {prompt: index for index, prompt in enumerate(labels)}
    xs = [row[0] for row in loaded]
    ys = [np.asarray([label_index[prompt] for prompt in row[1]])
          for row in loaded]
    results = [nested_target(xs, ys, target) for target in range(8)]
    if tuple(results[7]["selected_programs"]["mean"]) != (1, 3, 5):
        raise ValueError("nested target-008 mean rule differs from frozen run")
    summary = {}
    for rule in (*RULES, "fixed", "random_seed42", "all_six"):
        correct = [item["outer_correct_out_of_ten"][rule]
                   for item in results]
        regret = [item["posthoc_oracle_correct_out_of_ten"] - value
                  for item, value in zip(results, correct)]
        summary[rule] = {
            "correct_by_session": correct,
            "total_correct_out_of_80": sum(correct),
            "mean_correct_out_of_ten": float(np.mean(correct)),
            "mean_posthoc_regret_prompts": float(np.mean(regret)),
            "sessions_beating_fixed": sum(
                value > item["outer_correct_out_of_ten"]["fixed"]
                for item, value in zip(results, correct)),
        }
    payload = {
        "schema": "openalterego.emg-uka-cross-session-rules.v1",
        "scope": "Retrospective eight-session closed-prompt audible EMG; speaker 008; outer target excluded from its own selection, but all sessions now exposed; virtual recorded-channel subsets only",
        "source_download": manifest["download"],
        "archive_sha256": manifest["downloaded_archive_sha256"],
        "session_content_digests": {name: row[2] for name, row in
                                    zip(SESSIONS, loaded)},
        "speaker": "008", "sessions": SESSIONS,
        "channels": CHANNELS, "program_size": 3,
        "rules": {"mean": "inner mean, tie by inner minimum then channels",
                  "mean_minus_std": "inner mean minus population standard deviation, tie by mean then channels",
                  "worst_two_mean": "mean of the two worst inner sessions, tie by mean then channels",
                  "minimum": "worst inner session, tie by mean then channels"},
        "inner_folds_per_target": 7,
        "program_fits_per_target": 7 * len(CHANNELS),
        "outer_samples_per_target": 10,
        "summary": summary,
        "targets": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
