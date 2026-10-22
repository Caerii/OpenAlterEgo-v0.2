"""Leave-one-person-out calibration of mixed-program future-day rankings."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from openalterego.research.dense_programs import prepare_cv_folds, prepare_fold, score_programs
from openalterego.research.donor_calibration import nested_policy_gate, predict_left_out_person
from openalterego.research.hyser_features import site_coordinates
from openalterego.research.supported_programs import prepare_supported_fold
from openalterego.research.uniform_program_pool import uniform_program_prefix
from scripts.hyser_compositional_search import edge_mode_options
from scripts.hyser_mixed_program_pilot import load_mixed, physical_contacts
from scripts.hyser_program_search import fixed_grid_design
from scripts.hyser_two_support_robust_search import two_support_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mono-cache", type=Path, required=True)
    parser.add_argument("--pair-cache", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--candidates", type=int, default=5000)
    parser.add_argument("--backend", choices=("numpy", "rust"), default="rust")
    parser.add_argument("--nested-gate", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.output.exists() or len(args.subjects) < (4 if args.nested_gate else 3) or
            len(set(args.subjects)) != len(args.subjects) or
            args.candidates < 2):
        parser.error("new output, three unique people and at least two candidates required")
    start = time.perf_counter()
    programs = None
    expected_sites = expected_pairs = None
    proxies, outcomes, fixed_outcomes = [], [], []
    sources, support_indices = {}, {}
    for subject in args.subjects:
        day1, meta1 = load_mixed(args.mono_cache, args.pair_cache, subject, 1)
        day2, meta2 = load_mixed(args.mono_cache, args.pair_cache, subject, 2)
        if (meta1["site_names"] != meta2["site_names"] or
                meta1["pairs"] != meta2["pairs"] or
                meta1["source_record_sha256"] == meta2["source_record_sha256"]):
            raise ValueError("day pair does not align")
        if programs is None:
            expected_sites, expected_pairs = meta1["site_names"], meta1["pairs"]
            programs = uniform_program_prefix(edge_mode_options(meta1),
                                              args.candidates, 42)
            fixed = np.asarray([fixed_grid_design(
                site_coordinates(tuple(expected_sites)))], dtype=np.int16)
        elif (meta1["site_names"] != expected_sites or
              meta1["pairs"] != expected_pairs):
            raise ValueError("program grammar differs between people")
        first, second, support_ids, query_ids = two_support_split(day2[1])
        cv = score_programs(prepare_cv_folds(*day1, features_per_site=2),
                            programs, features_per_site=2, backend=args.backend)
        transfers = [score_programs(
            [prepare_fold(day1, (day2[0][ids], day2[1][ids]),
                          features_per_site=2)], programs,
            features_per_site=2, backend=args.backend) for ids in (first, second)]
        proxies.append(np.column_stack((cv, np.mean(transfers, axis=0))))
        fold = prepare_supported_fold(
            day1, (day2[0][support_ids], day2[1][support_ids]),
            (day2[0][query_ids], day2[1][query_ids]),
            features_per_site=2, support_weight=4.0)
        outcomes.append(score_programs([fold], programs,
                                       features_per_site=2, backend=args.backend))
        fixed_outcomes.append(float(score_programs(
            [fold], fixed, features_per_site=2, backend=args.backend)[0]))
        sources[subject] = [meta1["source_record_sha256"],
                            meta2["source_record_sha256"]]
        support_indices[subject] = {"support": support_ids.tolist(),
                                    "query": query_ids.tolist()}
        print("scored", subject, flush=True)
    proxies, outcomes = np.asarray(proxies), np.asarray(outcomes)
    rows = {}
    for target, subject in enumerate(args.subjects):
        learned, model = predict_left_out_person(proxies, outcomes, target)
        donor_prior = model.pop("donor_prior")
        baseline = .5 * (proxies[target, :, 0] + proxies[target, :, 1])
        chosen = {"two_support": int(np.argmax(baseline)),
                  "donor_only": int(np.argmax(donor_prior)),
                  "donor_calibrated": int(np.argmax(learned))}
        gate = nested_policy_gate(proxies, outcomes, target) if args.nested_gate else None
        if gate is not None:
            chosen["nested_gate"] = chosen[gate["selected_policy"]]
        rows[subject] = {
            "selection": {name: {
                "candidate_index": index,
                "program": programs[index].tolist(),
                "physical_contacts": physical_contacts(
                    programs[index], expected_pairs),
                "query_balanced_accuracy": float(outcomes[target, index])}
                for name, index in chosen.items()},
            "fixed_eight_site_query_balanced_accuracy": fixed_outcomes[target],
            "candidate_median_query": float(np.median(outcomes[target])),
            "candidate_query_oracle": float(outcomes[target].max()),
            "learned_rank_spearman": float(spearmanr(learned, outcomes[target]).statistic),
            "baseline_rank_spearman": float(spearmanr(baseline, outcomes[target]).statistic),
            "model": model,
        }
        if gate is not None:
            rows[subject]["nested_gate_evidence"] = gate
        print(subject, {name: round(row["query_balanced_accuracy"], 3)
                        for name, row in rows[subject]["selection"].items()},
              flush=True)
    result = {
        "schema": ("openalterego.hyser-nested-donor-gate.v1" if args.nested_gate
                   else "openalterego.hyser-donor-calibrated-program-search.v1"),
        "scope": "Retrospective exposed forearm subjects; outer target query labels excluded from donor calibration and candidate selection",
        "subjects": args.subjects, "candidate_count": args.candidates,
        "candidate_seed": 42, "backend": args.backend,
        "candidate_pool_sha256": hashlib.sha256(programs.tobytes()).hexdigest(),
        "donor_model": f"Within-person centered ridge on day-one CV, mean of two support transfer scores, and cross-fitted donor query mean; alpha=100; {len(args.subjects) - 1} donors per target",
        "sources": sources, "splits": support_indices,
        "results": rows, "elapsed_seconds": time.perf_counter() - start,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
