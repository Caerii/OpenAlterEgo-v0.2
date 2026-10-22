"""Compare guided and uniform program search at matched evaluation budgets."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from openalterego.research.dense_programs import prepare_cv_folds, prepare_fold, score_programs
from openalterego.research.guided_program_search import guided_search
from openalterego.research.supported_programs import prepare_supported_fold
from scripts.hyser_compositional_search import edge_mode_options
from scripts.hyser_mixed_program_pilot import load_mixed, physical_contacts
from scripts.hyser_support_conditioned_search import sample_programs
from scripts.hyser_two_support_robust_search import two_support_split


def analyze_subject(mono_cache: Path, pair_cache: Path, subject: str,
                    budget: int) -> dict:
    day1, meta1 = load_mixed(mono_cache, pair_cache, subject, 1)
    day2, meta2 = load_mixed(mono_cache, pair_cache, subject, 2)
    if (meta1["site_names"] != meta2["site_names"] or
            meta1["pairs"] != meta2["pairs"] or
            meta1["source_record_sha256"] == meta2["source_record_sha256"]):
        raise ValueError("day pair does not align")
    first, second, support_ids, query_ids = two_support_split(day2[1])
    cv_folds = prepare_cv_folds(*day1, features_per_site=2)
    transfer_folds = [prepare_fold(day1, (day2[0][ids], day2[1][ids]),
                                   features_per_site=2)
                      for ids in (first, second)]

    def selection_score(programs):
        cv = score_programs(cv_folds, programs, features_per_site=2)
        transfer = [score_programs([fold], programs, features_per_site=2)
                    for fold in transfer_folds]
        return .5 * cv + .25 * (transfer[0] + transfer[1])

    options = edge_mode_options(meta1)
    guided_start = time.perf_counter()
    guided, guided_scores, guided_winners = guided_search(
        options, selection_score, budget=budget, seed=42)
    guided_seconds = time.perf_counter() - guided_start
    random_start = time.perf_counter()
    random = sample_programs(options, budget, 42)
    random = random[np.random.default_rng(991).permutation(len(random))]
    random_scores = selection_score(random)
    random_seconds = time.perf_counter() - random_start
    budgets = [count for count in (500, 1000, 3000, 5000)
               if count <= budget]
    winners = {"guided": {count: guided_winners[count] for count in budgets},
               "random": {count: int(np.argmax(random_scores[:count]))
                          for count in budgets}}
    support = (day2[0][support_ids], day2[1][support_ids])
    query = (day2[0][query_ids], day2[1][query_ids])
    fold = prepare_supported_fold(day1, support, query,
                                  features_per_site=2, support_weight=4.0)
    selected = [(kind, count, index) for kind, by_budget in winners.items()
                for count, index in by_budget.items()]
    programs = np.asarray([guided[index] if kind == "guided" else random[index]
                           for kind, _, index in selected], dtype=np.int16)
    query_scores = score_programs([fold], programs, features_per_site=2)
    return {
        "day1_source_record_sha256": meta1["source_record_sha256"],
        "day2_source_record_sha256": meta2["source_record_sha256"],
        "support_indices": support_ids.tolist(),
        "query_indices": query_ids.tolist(),
        "search_seconds": {"guided": guided_seconds, "random": random_seconds},
        "outcomes": {
            kind: {str(count): {
                "program": programs[i].tolist(),
                "physical_contacts": physical_contacts(programs[i], meta1["pairs"]),
                "selection_score": float(guided_scores[index] if kind == "guided"
                                         else random_scores[index]),
                "query_balanced_accuracy": float(query_scores[i]),
            } for i, (chosen_kind, count, index) in enumerate(selected)
                if chosen_kind == kind}
            for kind in winners},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mono-cache", type=Path, required=True)
    parser.add_argument("--pair-cache", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--budget", type=int, default=5000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.budget < 500 or args.budget % 500:
        parser.error("new output and budget divisible by 500, at least 500, required")
    people = {}
    for subject in args.subjects:
        people[subject] = analyze_subject(args.mono_cache, args.pair_cache,
                                          subject, args.budget)
        print(subject, {kind: people[subject]["outcomes"][kind][str(args.budget)]
                        ["query_balanced_accuracy"] for kind in ("guided", "random")},
              flush=True)
    result = {
        "schema": "openalterego.hyser-guided-two-support-search.v1",
        "scope": "Retrospective exposed forearm subjects; digital measured-signal replay, not analog or facial hardware validation",
        "selection": "half day-one three-fold CV, one-quarter each independent day-two support repetition; query excluded",
        "comparison": "same complete-program evaluation budgets; random pool seed 42, randomly ordered with seed 991; guided search seed 42, batches of 500, 32 elites",
        "budget": args.budget,
        "subjects": people,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
