"""Freeze support-conditioned design choices before scoring query trials."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from openalterego.research import dense_programs, supported_programs
from openalterego.research.dense_programs import (
    prepare_cv_folds, prepare_fold, score_programs)
from openalterego.research.hyser_features import site_coordinates
from openalterego.research.hyser_fewshot import support_query
from openalterego.research.supported_programs import prepare_supported_fold
from scripts import hyser_compositional_search, hyser_mixed_program_pilot
from scripts.hyser_mixed_program_pilot import fixed_mixed, load_mixed, physical_contacts
from scripts.hyser_program_search import fixed_grid_design
from scripts.hyser_support_conditioned_search import sample_programs


SUBJECTS = ("06", "07", "08", "17", "18", "19", "20")
CODE_FILES = (Path(__file__), Path(dense_programs.__file__),
              Path(supported_programs.__file__),
              Path(hyser_compositional_search.__file__),
              Path(hyser_mixed_program_pilot.__file__),
              Path(sample_programs.__code__.co_filename))
CHOICES = ("half_cv_half_support", "day1_cv", "day1_to_support",
           "fixed_eight_site", "fixed_mixed",
           "half_cv_half_support_same_contacts")


def canonical_hash(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def code_hash() -> str:
    return hashlib.sha256(b"".join(path.read_bytes() for path in CODE_FILES)).hexdigest()


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
    result = {
        "schema": "openalterego.hyser-support-search-selection.v1",
        "scope": "Method-specific forearm digital replay; these subjects and day-two recordings exposed in earlier work",
        "code_sha256": code_hash(), "subjects": SUBJECTS,
        "candidate_count": 5000, "candidate_seed": 42,
        "support_shots_per_class": 1, "query_support_weight": 4.0,
        "primary": "half day-one CV plus half day-one-to-session-support score versus established fixed eight-site, same query",
        "success_rule": "mean paired gain >=0.05, >=5/7 people improve, no loss worse than -0.05",
        "choices": {},
    }
    for subject in SUBJECTS:
        day1, meta1 = load_mixed(mono_cache, pair_cache, subject, 1)
        day2, meta2 = load_mixed(mono_cache, pair_cache, subject, 2)
        if (meta1["site_names"] != meta2["site_names"] or
                meta1["pairs"] != meta2["pairs"]):
            raise ValueError(f"session metadata changed: {subject}")
        support_ids, _ = support_query(day2[1], 1)
        support = (day2[0][support_ids], day2[1][support_ids])
        options = hyser_compositional_search.edge_mode_options(meta1)
        candidates = sample_programs(options, 5000, 42)
        cv = score_programs(prepare_cv_folds(*day1, features_per_site=2),
                            candidates, features_per_site=2)
        transfer = score_programs(
            [prepare_fold(day1, support, features_per_site=2)],
            candidates, features_per_site=2)
        selected = {
            "half_cv_half_support": candidates[np.argmax((cv + transfer) / 2)],
            "day1_cv": candidates[np.argmax(cv)],
            "day1_to_support": candidates[np.argmax(transfer)],
        }
        selected["fixed_eight_site"] = fixed_grid_design(
            site_coordinates(tuple(meta1["site_names"])))
        selected["fixed_mixed"] = fixed_mixed(meta1)
        selected["half_cv_half_support_same_contacts"] = physical_contacts(
            selected["half_cv_half_support"], meta1["pairs"])
        result["choices"][subject] = {
            "day1_source_record_sha256": meta1["source_record_sha256"],
            "day2_source_record_sha256": meta2["source_record_sha256"],
            "pair_map_sha256": canonical_hash(
                {"sites": meta1["site_names"], "pairs": meta1["pairs"]}),
            "support_trials": len(support_ids),
            "programs": {name: [int(channel) for channel in selected[name]]
                         for name in CHOICES},
            "selected_day1_cv": float(cv[np.argmax((cv + transfer) / 2)]),
            "selected_day1_to_support": float(
                transfer[np.argmax((cv + transfer) / 2)]),
        }
        print(subject, "selected", result["choices"][subject]["programs"][
            "half_cv_half_support"], flush=True)
    write_exclusive(output, result)
    print("selection_sha256", canonical_hash(result), flush=True)


def score(mono_cache: Path, pair_cache: Path, selection_path: Path, output: Path):
    if output.exists():
        raise ValueError("score already exists")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if (selection.get("schema") != "openalterego.hyser-support-search-selection.v1" or
            selection.get("code_sha256") != code_hash() or
            tuple(selection.get("subjects", ())) != SUBJECTS):
        raise ValueError("selection, subject set, or code changed")
    result = {"schema": "openalterego.hyser-support-search-score.v1",
              "selection_sha256": canonical_hash(selection), "subjects": {}}
    for subject in SUBJECTS:
        day1, meta1 = load_mixed(mono_cache, pair_cache, subject, 1)
        day2, meta2 = load_mixed(mono_cache, pair_cache, subject, 2)
        recorded = selection["choices"][subject]
        for meta, source in ((meta1, "day1_source_record_sha256"),
                             (meta2, "day2_source_record_sha256")):
            if (meta["source_record_sha256"] != recorded[source] or
                    canonical_hash({"sites": meta["site_names"],
                                    "pairs": meta["pairs"]}) !=
                    recorded["pair_map_sha256"]):
                raise ValueError(f"source identity changed: {subject}")
        support_ids, query_ids = support_query(day2[1], 1)
        if len(support_ids) != recorded["support_trials"]:
            raise ValueError(f"support trial count changed: {subject}")
        fold = prepare_supported_fold(
            day1, (day2[0][support_ids], day2[1][support_ids]),
            (day2[0][query_ids], day2[1][query_ids]),
            features_per_site=2, support_weight=4.0)
        designs = np.asarray([recorded["programs"][name] for name in CHOICES],
                             dtype=np.int16)
        values = score_programs([fold], designs, features_per_site=2)
        scores = {name: float(value) for name, value in zip(CHOICES, values)}
        result["subjects"][subject] = {
            "support_trials": len(support_ids), "query_trials": len(query_ids),
            "query_balanced_accuracy": scores,
            "primary_paired_gain": (scores["half_cv_half_support"] -
                                    scores["fixed_eight_site"]),
        }
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
