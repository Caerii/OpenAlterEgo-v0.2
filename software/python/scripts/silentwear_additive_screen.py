"""Budget-matched compositional screen of recorded SilentWear channels."""

import argparse
import hashlib
import itertools
import json
import os
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from openalterego.research.compositional_surrogate import fit_binary_program_model
from openalterego.research.dense_programs import prepare_fold, score_programs_by_fold
from openalterego.research.program_search import normalize_batches
from scripts.silentwear_program_search import subject_sessions


NORMALIZATIONS = ("none", "recording_robust")
SUBSETS = np.asarray(list(itertools.combinations(range(14), 6)), dtype=np.int16)
PROGRAMS = np.tile(SUBSETS, (len(NORMALIZATIONS), 1))
MODES = np.repeat(np.arange(len(NORMALIZATIONS)), len(SUBSETS))
FEATURES = np.zeros((len(PROGRAMS), 15), dtype=np.float64)
FEATURES[np.arange(len(PROGRAMS))[:, None], PROGRAMS] = 1
FEATURES[:, 14] = MODES


def _scores(indices: np.ndarray, folds: tuple) -> np.ndarray:
    result = np.empty(len(indices), dtype=np.float64)
    for mode in range(len(NORMALIZATIONS)):
        positions = np.flatnonzero(MODES[indices] == mode)
        if len(positions):
            directional = score_programs_by_fold(
                folds[mode], PROGRAMS[indices[positions]],
                features_per_site=2, backend="rust")
            result[positions] = directional.min(axis=0)
    return result


def _freeze(path: Path, record: dict) -> str:
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":"),
                           allow_nan=False).encode()
    encoded = (json.dumps(record, sort_keys=True, indent=2,
                          allow_nan=False) + "\n").encode()
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != record:
            raise ValueError(f"selection record differs: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    return hashlib.sha256(canonical).hexdigest()


def analyze(root: Path, cache: Path, subject: str, seed: int,
            seed_exact: int, finalists_exact: int,
            selection_path: Path) -> dict:
    load_session, provenance, manifest_hash = subject_sessions(root, subject, cache)
    first, second = load_session(1), load_session(2)
    folds = []
    for norm in NORMALIZATIONS:
        day1, day2 = (normalize_batches(batches, norm)
                      for batches in (first, second))
        folds.append((prepare_fold(day1, day2, features_per_site=2),
                      prepare_fold(day2, day1, features_per_site=2)))
    order = np.random.default_rng(seed).permutation(len(PROGRAMS))
    uniform_count = seed_exact + finalists_exact
    seed_indices = order[:seed_exact]
    control_indices = order[seed_exact:uniform_count]
    start = time.perf_counter()
    seed_scores = _scores(seed_indices, folds)
    control_scores = _scores(control_indices, folds)
    exact_seconds = time.perf_counter() - start
    held_start = seed_exact - max(1, seed_exact // 5)
    diagnostic = fit_binary_program_model(
        FEATURES[seed_indices[:held_start]], seed_scores[:held_start])
    held_prediction = diagnostic.predict(FEATURES[seed_indices[held_start:]])
    model = fit_binary_program_model(FEATURES[seed_indices], seed_scores)
    start = time.perf_counter()
    remaining = order[uniform_count:]
    predicted = model.predict(FEATURES[remaining])
    finalists = remaining[np.lexsort((remaining, -predicted))[:finalists_exact]]
    screening_seconds = time.perf_counter() - start
    start = time.perf_counter()
    finalist_scores = _scores(finalists, folds)
    finalist_seconds = time.perf_counter() - start
    control_local = int(np.argmax(np.r_[seed_scores, control_scores]))
    control_index = int(order[control_local])
    seed_local = int(np.argmax(seed_scores))
    finalist_local = int(np.argmax(finalist_scores))
    screened_index = (int(seed_indices[seed_local]) if
                      seed_scores[seed_local] >= finalist_scores[finalist_local]
                      else int(finalists[finalist_local]))
    score_lookup = {int(index): float(score) for indices, scores in
                    ((seed_indices, seed_scores),
                     (control_indices, control_scores),
                     (finalists, finalist_scores))
                    for index, score in zip(indices, scores)}
    choices = {"uniform_exact": control_index,
               "additive_screened": screened_index}
    selection = {name: {"candidate_index": index,
                        "channels": PROGRAMS[index].tolist(),
                        "normalization": NORMALIZATIONS[MODES[index]],
                        "selection_score": score_lookup[index]}
                 for name, index in choices.items()}
    receipt = _freeze(selection_path, {
        "schema": "openalterego.silentwear-additive-selection.v1",
        "subject": subject, "manifest_sha256": manifest_hash,
        "seed": seed, "seed_exact": seed_exact,
        "finalists_exact": finalists_exact,
        "selection": selection,
        "scope": "Local prequery record; retrospective exposed sessions, not independent preregistration"})

    third = load_session(3)
    for name, index in choices.items():
        norm = NORMALIZATIONS[MODES[index]]
        fit_x, fit_y = normalize_batches(first + second, norm)
        query_x, query_y = normalize_batches(third, norm)
        columns = [column for channel in PROGRAMS[index]
                   for column in (2 * int(channel), 2 * int(channel) + 1)]
        classifier = make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, max_iter=2000, class_weight="balanced", random_state=42))
        classifier.fit(fit_x[:, columns], fit_y)
        selection[name]["query_balanced_accuracy"] = float(
            balanced_accuracy_score(query_y,
                                    classifier.predict(query_x[:, columns])))
    return {"manifest_sha256": manifest_hash,
            "selection_receipt_sha256": receipt,
            "selection_record": str(selection_path),
            "source_batches": provenance,
            "heldout_seed_spearman": float(spearmanr(
                held_prediction, seed_scores[held_start:]).statistic),
            "heldout_seed_mae": float(np.mean(np.abs(
                held_prediction - seed_scores[held_start:]))),
            "times_seconds": {"initial_exact_both_policies": exact_seconds,
                              "screen_remaining": screening_seconds,
                              "finalist_exact": finalist_seconds},
            "selection": selection}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", default=["S01", "S02", "S03"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-exact", type=int, default=150)
    parser.add_argument("--finalists-exact", type=int, default=150)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.output.exists() or
            not set(args.subjects).issubset({"S01", "S02", "S03"}) or
            args.seed_exact < 10 or args.finalists_exact < 1 or
            args.seed_exact + args.finalists_exact >= len(PROGRAMS)):
        parser.error("new output, development subjects and valid budget required")
    rows = {}
    for subject in args.subjects:
        selection_path = args.output.with_name(
            f"{args.output.stem}-{subject}-selection.json")
        rows[subject] = analyze(args.root, args.cache_dir, subject, args.seed,
                                args.seed_exact, args.finalists_exact,
                                selection_path)
        print(subject, {name: round(value["query_balanced_accuracy"], 3)
                        for name, value in rows[subject]["selection"].items()},
              flush=True)
    result = {
        "schema": "openalterego.silentwear-additive-screen.v1",
        "scope": "Retrospective exposed silent-command sessions; virtual subsets of 14 recorded channels only; day 3 loaded after local choice records",
        "seed": args.seed, "seed_exact": args.seed_exact,
        "finalists_exact": args.finalists_exact,
        "uniform_exact": args.seed_exact + args.finalists_exact,
        "program_count": len(PROGRAMS),
        "selection": "minimum of session1-to-2 and session2-to-1 balanced ridge accuracy, alpha=10",
        "final_decoder": "balanced C=1 logistic fit on sessions 1+2, evaluated on session 3",
        "subjects": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
