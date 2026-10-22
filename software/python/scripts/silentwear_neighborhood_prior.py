"""Test one-swap score smoothing for future-session virtual-montage choice."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from openalterego.research import neighborhood_prior
from openalterego.research.neighborhood_prior import one_swap_smoothed_scores
from openalterego.research.program_search import normalize_batches, ridge_proxy_rank
from scripts.silentwear_program_search import subject_sessions


def freeze(path: Path, subject: str, manifest_sha256: str, choices: dict) -> str:
    record = {"schema": "openalterego.one-swap-selection.v1",
              "subject": subject, "manifest_sha256": manifest_sha256,
              "code_sha256": hashlib.sha256(
                  Path(__file__).read_bytes() +
                  Path(neighborhood_prior.__file__).read_bytes()).hexdigest(),
              "choices": choices,
              "scope": "Local exclusive record before loading session 3; not external preregistration."}
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(canonical).hexdigest()
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != record:
            raise ValueError(f"selection record differs: {path}")
        return digest
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        json.dump(record, output, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    return digest


def score_choices(first, second):
    forward, reverse = {}, {}
    for norm in ("none", "recording_robust"):
        train = normalize_batches(first, norm)
        validation = normalize_batches(second, norm)
        forward.update({(norm, program.channels): score
                        for program, score in ridge_proxy_rank(train, validation, norm)})
        reverse.update({(norm, program.channels): score
                        for program, score in ridge_proxy_rank(validation, train, norm)})
    programs = sorted(forward)
    if forward.keys() != reverse.keys() or len(programs) != 6006:
        raise ValueError("incomplete paired proxy scores")
    a = np.asarray([forward[program] for program in programs])
    b = np.asarray([reverse[program] for program in programs])
    scores = {"raw_bidirectional": np.minimum(a, b),
              "one_swap_smoothed": np.minimum(
                  one_swap_smoothed_scores(programs, a),
                  one_swap_smoothed_scores(programs, b))}
    choices = {}
    for name, values in scores.items():
        index = int(np.argmax(values))
        choices[name] = {"normalization": programs[index][0],
                         "channels": list(programs[index][1]),
                         "selection_proxy_score": float(values[index])}
    return choices


def score_logistic(choice, train_batches, test_batches):
    norm = choice["normalization"]
    train_x, train_y = normalize_batches(train_batches, norm)
    test_x, test_y = normalize_batches(test_batches, norm)
    columns = [column for channel in choice["channels"]
               for column in (2 * channel, 2 * channel + 1)]
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=1, max_iter=2000, class_weight="balanced", random_state=42))
    model.fit(train_x[:, columns], train_y)
    return float(balanced_accuracy_score(test_y, model.predict(test_x[:, columns])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", default=["S01", "S02", "S03"])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not set(args.subjects).issubset({"S01", "S02", "S03"}):
        parser.error("S04 is excluded from this development diagnostic")
    config = {"subjects": args.subjects, "proxy": "class-balanced ridge alpha 10",
              "choice_rules": ["raw_bidirectional", "one_swap_smoothed"],
              "decoder": "class-balanced logistic C=1 on sessions 1+2"}
    if args.output.exists():
        if not args.resume:
            parser.error("output exists; use --resume")
        result = json.loads(args.output.read_text(encoding="utf-8"))
        if result.get("schema") != "openalterego.one-swap-prior.v1" or result.get("config") != config:
            parser.error("existing run configuration differs")
    else:
        result = {"schema": "openalterego.one-swap-prior.v1", "config": config,
                  "scope": "Retrospective S01-S03 development; session 3 previously exposed; recorded-channel substitutions only.",
                  "subjects": {}}
    for subject in args.subjects:
        if subject in result["subjects"]:
            continue
        load_session, provenance, manifest_sha256 = subject_sessions(
            args.root, subject, args.cache_dir)
        first, second = load_session(1), load_session(2)
        start = time.perf_counter()
        choices = score_choices(first, second)
        selection_seconds = time.perf_counter() - start
        selection_path = args.output.with_name(f"{args.output.stem}-{subject}-selection.json")
        selection_sha256 = freeze(selection_path, subject, manifest_sha256, choices)
        third = load_session(3)
        for choice in choices.values():
            choice["test_balanced_accuracy"] = score_logistic(choice, first + second, third)
        result["subjects"][subject] = {
            "choices": choices, "selection_wall_seconds": selection_seconds,
            "input_manifest_sha256": manifest_sha256,
            "selection_record": str(selection_path),
            "selection_sha256": selection_sha256, "inputs": provenance}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f"{args.output.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                             encoding="utf-8")
        os.replace(temporary, args.output)
        print(subject, {name: row["test_balanced_accuracy"] for name, row in choices.items()},
              selection_seconds, flush=True)


if __name__ == "__main__":
    main()
