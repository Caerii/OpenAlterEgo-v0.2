"""Freeze day-one alpha choices before a new-person few-shot Hyser test."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from openalterego.research import dense_programs, hyser_features, hyser_fewshot
from openalterego.research.dense_programs import prepare_cv_folds, score_programs
from openalterego.research.hyser_features import site_coordinates
from openalterego.research.hyser_fewshot import ridge_query_score, support_query
from scripts import hyser_program_search, hyser_temporal_fixed_pilot
from scripts.hyser_program_search import fixed_grid_design
from scripts.hyser_temporal_fixed_pilot import load


SCHEMA = "openalterego.hyser-fewshot-holdout-selection.v1"
ALPHAS = (1.0, 10.0, 100.0, 1000.0)
CODE_FILES = (Path(__file__), Path(dense_programs.__file__),
              Path(hyser_features.__file__), Path(hyser_fewshot.__file__),
              Path(hyser_program_search.__file__),
              Path(hyser_temporal_fixed_pilot.__file__))


def code_hash() -> str:
    return hashlib.sha256(b"".join(path.read_bytes() for path in CODE_FILES)).hexdigest()


def canonical_hash(record: dict) -> str:
    return hashlib.sha256(json.dumps(record, sort_keys=True,
                                     separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def write_exclusive(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def select(root: Path, cache: Path, subjects: list[str], output: Path):
    if output.exists():
        raise ValueError("selection output already exists")
    if any((root / f"pr_dataset/subject{subject}_session2").exists() or
           list(cache.glob(f"subject{subject}-session2-*.npz"))
           for subject in subjects):
        raise ValueError("next-day raw data or features already exist for cohort")
    result = {"schema": SCHEMA, "code_sha256": code_hash(),
              "protocol": {"primary": "one labeled support trial per day-two gesture, whole-trial fixed eight-site ridge, support weight 1 versus zero support on identical query trials",
                           "secondary": "two labeled support trials per gesture, weight 4",
                           "alpha_selection": "day-one 3-fold stratified balanced accuracy; choices 1,10,100,1000",
                           "support_split": "first recorded repetition per class as support; later repetitions of that class as query",
                           "scope": "episodic within-class replay, not globally chronological online calibration"},
              "subjects": subjects, "choices": {}}
    for subject in subjects:
        train, meta = load(cache, subject, 1, 2)
        sites = site_coordinates(tuple(meta["site_names"]))
        fixed = fixed_grid_design(sites)
        folds = prepare_cv_folds(*train, features_per_site=2)
        cv = {str(alpha): float(score_programs(
            folds, np.asarray([fixed]), features_per_site=2,
            alpha=alpha)[0]) for alpha in ALPHAS}
        alpha = min(ALPHAS, key=lambda value: (-cv[str(value)], value))
        result["choices"][subject] = {
            "train_source_record_sha256": meta["source_record_sha256"],
            "publisher_manifest_sha256": meta["publisher_manifest_sha256"],
            "site_names": meta["site_names"], "channels": fixed,
            "day_one_cv_by_alpha": cv, "chosen_alpha": alpha,
            "day_one_trials": len(train[1])}
        print(subject, "alpha", alpha, "cv", round(cv[str(alpha)], 3), flush=True)
    write_exclusive(output, result)
    print("selection_sha256", canonical_hash(result), flush=True)


def score(cache: Path, selection_path: Path, output: Path):
    if output.exists():
        raise ValueError("score output already exists")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("schema") != SCHEMA or selection.get("code_sha256") != code_hash():
        raise ValueError("frozen selection code differs")
    result = {"schema": "openalterego.hyser-fewshot-holdout-score.v1",
              "selection_sha256": canonical_hash(selection),
              "protocol": selection["protocol"], "subjects": {}}
    for subject in selection["subjects"]:
        train, train_meta = load(cache, subject, 1, 2)
        target, target_meta = load(cache, subject, 2, 2)
        choice = selection["choices"][subject]
        if (train_meta["source_record_sha256"] != choice["train_source_record_sha256"] or
                target_meta["publisher_manifest_sha256"] != choice["publisher_manifest_sha256"] or
                target_meta["site_names"] != choice["site_names"]):
            raise ValueError("data source or channel order differs from frozen choice")
        by_shots = {}
        for shots, adapted_weight in ((1, 1.0), (2, 4.0)):
            support, query = support_query(target[1], shots)
            scores = {str(weight): ridge_query_score(
                train, target, choice["channels"], features_per_site=2,
                alpha=choice["chosen_alpha"], support=support, query=query,
                support_weight=weight)
                for weight in (0.0, adapted_weight)}
            by_shots[str(shots)] = {"support_trials": len(support),
                                   "query_trials": len(query),
                                   "query_balanced_accuracy": scores,
                                   "paired_gain": scores[str(adapted_weight)] - scores["0.0"]}
        result["subjects"][subject] = {
            "test_source_record_sha256": target_meta["source_record_sha256"],
            "shots": by_shots}
        print(subject, "primary_gain", round(by_shots["1"]["paired_gain"], 3), flush=True)
    write_exclusive(output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    choose = sub.add_parser("select")
    choose.add_argument("--root", type=Path, required=True)
    choose.add_argument("--cache-dir", type=Path, required=True)
    choose.add_argument("--subjects", nargs="+", required=True)
    choose.add_argument("--output", type=Path, required=True)
    measure = sub.add_parser("score")
    measure.add_argument("--cache-dir", type=Path, required=True)
    measure.add_argument("--selection", type=Path, required=True)
    measure.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "select":
        if len(set(args.subjects)) != len(args.subjects) or any(
                subject not in {f"{index:02d}" for index in range(1, 21)}
                for subject in args.subjects):
            parser.error("invalid or repeated subject")
        select(args.root, args.cache_dir, args.subjects, args.output)
    else:
        score(args.cache_dir, args.selection, args.output)


if __name__ == "__main__":
    main()
