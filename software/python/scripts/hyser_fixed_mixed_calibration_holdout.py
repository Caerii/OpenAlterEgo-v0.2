"""Freeze and score a fixed equal-contact measurement-mode calibration test."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from openalterego.research import dense_programs, supported_programs
from openalterego.research.dense_programs import score_programs
from openalterego.research.hyser_features import site_coordinates
from openalterego.research.hyser_fewshot import support_query
from openalterego.research.supported_programs import prepare_supported_fold
from scripts import hyser_mixed_program_pilot
from scripts.hyser_mixed_program_pilot import (
    fixed_mixed, load_mixed, physical_contacts)
from scripts.hyser_program_search import fixed_grid_design


SUBJECTS = ("06", "07", "08", "17", "18", "19", "20")
CODE_FILES = (Path(__file__), Path(dense_programs.__file__),
              Path(supported_programs.__file__),
              Path(hyser_mixed_program_pilot.__file__))


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


def select(mono_cache: Path, pair_cache: Path, output: Path):
    if output.exists():
        raise ValueError("selection already exists")
    result = {"schema": "openalterego.hyser-fixed-mixed-calibration-selection.v1",
              "code_sha256": code_hash(),
              "scope": "Method-specific evaluation only; subjects and day-two recordings exposed in other Hyser experiments",
              "subjects": SUBJECTS,
              "protocol": {
                  "primary": "one support trial per gesture, weight 4; fixed mixed minus established fixed eight-site shared-reference on identical query trials",
                  "success_rule": "mean paired gain >=0.05, at least 5/7 improve, no individual loss worse than -0.05",
                  "secondary": "support weight 1 and same-contact all-shared-reference comparator",
                  "alpha": 10.0, "support_split": "first recorded repetition per class; remaining repetitions query, not globally chronological",
                  "physical_cost": "eight array contacts plus a common reference; eight AFE channels"},
              "choices": {}}
    for subject in SUBJECTS:
        _, meta = load_mixed(mono_cache, pair_cache, subject, 1)
        mixed = fixed_mixed(meta)
        established = fixed_grid_design(site_coordinates(tuple(meta["site_names"])))
        same_contacts = physical_contacts(mixed, meta["pairs"])
        result["choices"][subject] = {
            "day1_source_record_sha256": meta["source_record_sha256"],
            "pair_map_sha256": canonical_hash(
                {"sites": meta["site_names"], "pairs": meta["pairs"]}),
            "fixed_mixed": mixed,
            "fixed_eight_site": established,
            "same_contacts_all_shared_reference": same_contacts}
        print(subject, "fixed", mixed, flush=True)
    write_exclusive(output, result)
    print("selection_sha256", canonical_hash(result), flush=True)


def score(mono_cache: Path, pair_cache: Path, selection_path: Path,
          output: Path):
    if output.exists():
        raise ValueError("score already exists")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if (selection.get("schema") !=
            "openalterego.hyser-fixed-mixed-calibration-selection.v1" or
            selection.get("code_sha256") != code_hash() or
            tuple(selection.get("subjects", ())) != SUBJECTS):
        raise ValueError("frozen selection or code differs")
    result = {"schema": "openalterego.hyser-fixed-mixed-calibration-score.v1",
              "selection_sha256": canonical_hash(selection),
              "protocol": selection["protocol"], "subjects": {}}
    names = ("fixed_mixed", "fixed_eight_site",
             "same_contacts_all_shared_reference")
    for subject in SUBJECTS:
        day1, meta1 = load_mixed(mono_cache, pair_cache, subject, 1)
        day2, meta2 = load_mixed(mono_cache, pair_cache, subject, 2)
        choice = selection["choices"][subject]
        if (meta1["source_record_sha256"] != choice["day1_source_record_sha256"] or
                canonical_hash({"sites": meta1["site_names"],
                                "pairs": meta1["pairs"]}) !=
                choice["pair_map_sha256"] or
                canonical_hash({"sites": meta2["site_names"],
                                "pairs": meta2["pairs"]}) !=
                choice["pair_map_sha256"]):
            raise ValueError(f"source or pair map differs: {subject}")
        designs = np.asarray([choice[name] for name in names], dtype=np.int16)
        support, query = support_query(day2[1], 1)
        scores = {}
        for weight in (0.0, 1.0, 4.0):
            fold = prepare_supported_fold(
                day1, (day2[0][support], day2[1][support]),
                (day2[0][query], day2[1][query]),
                features_per_site=2, support_weight=weight)
            values = score_programs([fold], designs, features_per_site=2)
            scores[str(weight)] = {name: float(value)
                                   for name, value in zip(names, values)}
        result["subjects"][subject] = {
            "day2_source_record_sha256": meta2["source_record_sha256"],
            "support_trials": len(support), "query_trials": len(query),
            "query_balanced_accuracy": scores,
            "primary_paired_gain": (scores["4.0"]["fixed_mixed"] -
                                    scores["4.0"]["fixed_eight_site"])}
        print(subject, "primary_gain", round(result["subjects"][subject][
            "primary_paired_gain"], 3), flush=True)
    write_exclusive(output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    choose = sub.add_parser("select")
    choose.add_argument("--mono-cache", type=Path, required=True)
    choose.add_argument("--pair-cache", type=Path, required=True)
    choose.add_argument("--output", type=Path, required=True)
    evaluate = sub.add_parser("score")
    evaluate.add_argument("--mono-cache", type=Path, required=True)
    evaluate.add_argument("--pair-cache", type=Path, required=True)
    evaluate.add_argument("--selection", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "select":
        select(args.mono_cache, args.pair_cache, args.output)
    else:
        score(args.mono_cache, args.pair_cache, args.selection, args.output)


if __name__ == "__main__":
    main()
