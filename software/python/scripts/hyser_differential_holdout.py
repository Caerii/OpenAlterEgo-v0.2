"""Freeze 5,000-pair-program choices before differential day-two extraction."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from openalterego.research import dense_programs
from openalterego.research.dense_programs import (
    prepare_cv_folds, prepare_fold, score_programs)
from openalterego.research.hyser_features import site_coordinates
from openalterego.research.hyser_search import ridge_test
from scripts import hyser_differential_program_pilot
from scripts.hyser_differential_program_pilot import (
    candidate_programs, fixed_program, load)
from scripts.hyser_program_search import fixed_grid_design, load_session


SUBJECTS = ("06", "07", "08", "17", "18", "19", "20")
CODE_FILES = (Path(__file__), Path(dense_programs.__file__),
              Path(hyser_differential_program_pilot.__file__))


def code_hash() -> str:
    return hashlib.sha256(b"".join(path.read_bytes() for path in CODE_FILES)).hexdigest()


def canonical_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def write_exclusive(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def select(cache: Path, output: Path):
    if output.exists() or any(list(cache.glob(f"subject{subject}-session2-*.npz"))
                              for subject in SUBJECTS):
        raise ValueError("selection exists or day-two differential cache already exposed")
    result = {"schema": "openalterego.hyser-differential-holdout-selection.v1",
              "code_sha256": code_hash(),
              "scope": "Method-specific held-out pair outcomes; people/day-two labels were exposed in other Hyser work",
              "subjects": SUBJECTS,
              "protocol": {"candidate_count": 5000, "candidate_seed": 42,
                           "day1_cv_seed": 42, "day1_cv_folds": 3,
                           "ridge_alpha": 10.0, "afe_channels": 8,
                           "active_contacts": 12,
                           "primary": "chosen minus fixed differential day-two balanced accuracy",
                           "success_rule": "mean paired gain >=0.05, at least 5/7 improve, no individual loss worse than -0.05, and chosen mean >= fixed eight-site monopolar mean"},
              "choices": {}}
    for subject in SUBJECTS:
        train, meta = load(cache, subject, 1)
        designs = candidate_programs(meta, 5000)
        cv = score_programs(prepare_cv_folds(*train, features_per_site=2),
                            designs, features_per_site=2)
        chosen = int(np.argmax(cv))
        result["choices"][subject] = {
            "day1_source_record_sha256": meta["source_record_sha256"],
            "pair_map_sha256": canonical_hash({"sites": meta["site_names"],
                                                "pairs": meta["pairs"]}),
            "selected_channels": designs[chosen].astype(int).tolist(),
            "selected_day1_cv": float(cv[chosen]),
            "fixed_differential_channels": fixed_program(meta)}
        print(subject, "cv", round(float(cv[chosen]), 3), flush=True)
    write_exclusive(output, result)
    print("selection_sha256", canonical_hash(result), flush=True)


def score(cache: Path, monopolar_cache: Path, selection_path: Path,
          output: Path):
    if output.exists():
        raise ValueError("score output already exists")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if (selection.get("schema") != "openalterego.hyser-differential-holdout-selection.v1"
            or selection.get("code_sha256") != code_hash()
            or tuple(selection.get("subjects", ())) != SUBJECTS):
        raise ValueError("frozen selection/code differs")
    result = {"schema": "openalterego.hyser-differential-holdout-score.v1",
              "selection_sha256": canonical_hash(selection),
              "protocol": selection["protocol"], "subjects": {}}
    for subject in SUBJECTS:
        day1, meta1 = load(cache, subject, 1)
        day2, meta2 = load(cache, subject, 2)
        choice = selection["choices"][subject]
        if (choice["day1_source_record_sha256"] != meta1["source_record_sha256"] or
                choice["pair_map_sha256"] != canonical_hash(
                    {"sites": meta2["site_names"], "pairs": meta2["pairs"]})):
            raise ValueError(f"source or pair map changed: {subject}")
        designs = np.asarray([choice["selected_channels"],
                              choice["fixed_differential_channels"]], dtype=np.int16)
        selected, fixed = score_programs(
            [prepare_fold(day1, day2, features_per_site=2)], designs,
            features_per_site=2)
        mono1, original1 = load_session(monopolar_cache, subject, 1)
        mono2, original2 = load_session(monopolar_cache, subject, 2)
        if (original1["source_record_sha256"] != meta1["source_record_sha256"] or
                original2["source_record_sha256"] != meta2["source_record_sha256"]):
            raise ValueError(f"monopolar source identity differs: {subject}")
        mono_fixed = ridge_test(mono1, mono2, fixed_grid_design(
            site_coordinates(tuple(original1["site_names"]))))
        result["subjects"][subject] = {
            "day2_source_record_sha256": meta2["source_record_sha256"],
            "selected_differential_day2": float(selected),
            "fixed_differential_day2": float(fixed),
            "fixed_monopolar_day2": float(mono_fixed),
            "selected_minus_fixed_differential": float(selected - fixed)}
        print(subject, "differential_gain", round(float(selected - fixed), 3),
              flush=True)
    write_exclusive(output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    choose = sub.add_parser("select")
    choose.add_argument("--cache-dir", type=Path, required=True)
    choose.add_argument("--output", type=Path, required=True)
    evaluate = sub.add_parser("score")
    evaluate.add_argument("--cache-dir", type=Path, required=True)
    evaluate.add_argument("--monopolar-cache", type=Path, required=True)
    evaluate.add_argument("--selection", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "select":
        select(args.cache_dir, args.output)
    else:
        score(args.cache_dir, args.monopolar_cache,
              args.selection, args.output)


if __name__ == "__main__":
    main()
