"""Freeze one-cell robust programs before shifted query scoring."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from openalterego.research import contact_shift, dense_programs, supported_programs
from openalterego.research.contact_shift import (
    SHIFTS, align_shifted_features, channel_shift_maps, robust_options)
from openalterego.research.dense_programs import (
    prepare_cv_folds, prepare_fold, score_programs)
from openalterego.research.hyser_features import site_coordinates
from openalterego.research.hyser_fewshot import support_query
from openalterego.research.supported_programs import prepare_supported_fold
from scripts import (hyser_compositional_search, hyser_contact_shift_search,
                     hyser_mixed_program_pilot, hyser_support_conditioned_search)
from scripts.hyser_compositional_search import edge_mode_options
from scripts.hyser_mixed_program_pilot import load_mixed
from scripts.hyser_program_search import fixed_grid_design
from scripts.hyser_support_conditioned_search import sample_programs


SUBJECTS = ("06", "07", "08", "17", "18", "19", "20")
NAMES = ("worst_shift_selected", "nominal_selected", "fixed_eight_site")
CODE_FILES = (Path(__file__), Path(contact_shift.__file__),
              Path(dense_programs.__file__), Path(supported_programs.__file__),
              Path(hyser_compositional_search.__file__),
              Path(hyser_contact_shift_search.__file__),
              Path(hyser_mixed_program_pilot.__file__),
              Path(hyser_support_conditioned_search.__file__))


def canonical_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


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
        "schema": "openalterego.hyser-contact-shift-selection.v1",
        "scope": "Method-specific forearm HD array replay, prior participant/day-two exposure",
        "code_sha256": code_hash(), "subjects": SUBJECTS,
        "shifts": SHIFTS, "candidate_count": 5000, "candidate_seed": 42,
        "support_shots_per_class": 1, "support_weight": 4.0,
        "primary": "worst-shift selected minus nominal selected, mean over five shift scenarios per person",
        "success_rule": "robust-nominal mean >=0.02, robust-fixed mean >=0.05, robust beats fixed for >=5/7, no robust-fixed loss below -0.05",
        "choices": {},
    }
    for subject in SUBJECTS:
        day1, meta1 = load_mixed(mono_cache, pair_cache, subject, 1)
        day2, meta2 = load_mixed(mono_cache, pair_cache, subject, 2)
        if (meta1["site_names"] != meta2["site_names"] or
                meta1["pairs"] != meta2["pairs"]):
            raise ValueError(f"session metadata differs: {subject}")
        maps = channel_shift_maps(meta1)
        options = robust_options(edge_mode_options(meta1), maps)
        candidates = sample_programs(options, 5000, 42)
        support_ids, _ = support_query(day2[1], 1)
        cv = score_programs(prepare_cv_folds(*day1, features_per_site=2),
                            candidates, features_per_site=2)
        transfer = []
        for channel_map in maps:
            shifted = align_shifted_features(day2[0], channel_map,
                                             features_per_channel=2)
            support = (shifted[support_ids], day2[1][support_ids])
            transfer.append(score_programs(
                [prepare_fold(day1, support, features_per_site=2)],
                candidates, features_per_site=2))
        objective = (cv[None, :] + np.asarray(transfer)) / 2
        nominal_index = int(np.argmax(objective[0]))
        robust_index = int(np.argmax(np.min(objective, axis=0)))
        programs = {
            "worst_shift_selected": tuple(map(int, candidates[robust_index])),
            "nominal_selected": tuple(map(int, candidates[nominal_index])),
            "fixed_eight_site": tuple(fixed_grid_design(
                site_coordinates(tuple(meta1["site_names"])))),
        }
        if np.any(maps[:, programs["fixed_eight_site"]] < 0):
            raise ValueError("fixed comparator crosses shift boundary")
        result["choices"][subject] = {
            "day1_source_record_sha256": meta1["source_record_sha256"],
            "day2_source_record_sha256": meta2["source_record_sha256"],
            "pair_map_sha256": canonical_hash({
                "sites": meta1["site_names"], "pairs": meta1["pairs"]}),
            "grammar_options_per_grid": tuple(len(group) for group in options),
            "support_trials": len(support_ids),
            "programs": programs,
            "nominal_selection_score": float(objective[0, nominal_index]),
            "robust_worst_selection_score": float(
                np.min(objective[:, robust_index])),
        }
        print(subject, "robust", programs["worst_shift_selected"], flush=True)
    write_exclusive(output, result)
    print("selection_sha256", canonical_hash(result), flush=True)


def score(mono_cache: Path, pair_cache: Path,
          selection_path: Path, output: Path):
    if output.exists():
        raise ValueError("score already exists")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if (selection.get("schema") != "openalterego.hyser-contact-shift-selection.v1" or
            selection.get("code_sha256") != code_hash() or
            tuple(selection.get("subjects", ())) != SUBJECTS or
            tuple(map(tuple, selection.get("shifts", ()))) != SHIFTS):
        raise ValueError("selection or code changed")
    result = {"schema": "openalterego.hyser-contact-shift-score.v1",
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
        maps = channel_shift_maps(meta1)
        designs = np.asarray([recorded["programs"][name] for name in NAMES],
                             dtype=np.int16)
        if np.any(maps[:, designs] < 0):
            raise ValueError("selected program crosses shift boundary")
        support_ids, query_ids = support_query(day2[1], 1)
        if len(support_ids) != recorded["support_trials"]:
            raise ValueError("support trial count changed")
        scores = {}
        for shift, channel_map in zip(SHIFTS, maps):
            shifted = align_shifted_features(day2[0], channel_map,
                                             features_per_channel=2)
            fold = prepare_supported_fold(
                day1, (shifted[support_ids], day2[1][support_ids]),
                (shifted[query_ids], day2[1][query_ids]),
                features_per_site=2, support_weight=4.0)
            values = score_programs([fold], designs, features_per_site=2)
            scores[str(shift)] = {name: float(value)
                                  for name, value in zip(NAMES, values)}
        averages = {name: float(np.mean([scores[str(shift)][name]
                                         for shift in SHIFTS]))
                    for name in NAMES}
        worst = {name: float(min(scores[str(shift)][name]
                                 for shift in SHIFTS))
                 for name in NAMES}
        result["subjects"][subject] = {
            "support_trials": len(support_ids), "query_trials": len(query_ids),
            "query_balanced_accuracy_by_shift": scores,
            "mean_over_shifts": averages, "worst_shift": worst,
            "robust_minus_nominal": (averages["worst_shift_selected"] -
                                     averages["nominal_selected"]),
            "robust_minus_fixed": (averages["worst_shift_selected"] -
                                   averages["fixed_eight_site"]),
        }
        print(subject, "robust-nominal", round(result["subjects"][subject][
            "robust_minus_nominal"], 3), "robust-fixed", round(
                result["subjects"][subject]["robust_minus_fixed"], 3),
              flush=True)
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
