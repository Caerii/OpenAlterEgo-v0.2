"""Budget-matched additive grammar screening versus uniform exact search."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from openalterego.research.compositional_surrogate import fit_additive_program_model
from openalterego.research.dense_programs import prepare_cv_folds, prepare_fold, score_programs
from openalterego.research.supported_programs import prepare_supported_fold
from openalterego.research.uniform_program_pool import uniform_program_prefix_with_codes
from scripts.hyser_compositional_search import edge_mode_options
from scripts.hyser_mixed_program_pilot import load_mixed, physical_contacts
from scripts.hyser_two_support_robust_search import two_support_split


def analyze_subject(mono_cache: Path, pair_cache: Path, subject: str,
                    programs: np.ndarray, codes: np.ndarray,
                    topology: tuple, option_sizes: tuple[int, ...],
                    *, train_count: int, finalist_count: int,
                    backend: str) -> dict:
    day1, meta1 = load_mixed(mono_cache, pair_cache, subject, 1)
    day2, meta2 = load_mixed(mono_cache, pair_cache, subject, 2)
    if (topology != (meta1["site_names"], meta1["pairs"]) or
            topology != (meta2["site_names"], meta2["pairs"]) or
            meta1["source_record_sha256"] == meta2["source_record_sha256"]):
        raise ValueError("subject data do not match shared grammar and day split")
    first, second, support_ids, query_ids = two_support_split(day2[1])
    cv_folds = prepare_cv_folds(*day1, features_per_site=2)
    transfer_folds = [prepare_fold(day1, (day2[0][ids], day2[1][ids]),
                                   features_per_site=2) for ids in (first, second)]

    def selection_scores(indices):
        batch = programs[indices]
        cv = score_programs(cv_folds, batch, features_per_site=2,
                            backend=backend)
        transfer = [score_programs([fold], batch, features_per_site=2,
                                   backend=backend)
                    for fold in transfer_folds]
        return .5 * cv + .25 * (transfer[0] + transfer[1])

    start = time.perf_counter()
    uniform_count = train_count + finalist_count
    seed_scores = selection_scores(slice(0, train_count))
    seed_seconds = time.perf_counter() - start
    start = time.perf_counter()
    baseline_extra = selection_scores(slice(train_count, uniform_count))
    baseline_extra_seconds = time.perf_counter() - start
    initial = np.r_[seed_scores, baseline_extra]
    validation_start = train_count - max(1, train_count // 5)
    fit_start = time.perf_counter()
    diagnostic = fit_additive_program_model(
        codes[:validation_start], initial[:validation_start], option_sizes)
    held_prediction = diagnostic.predict(codes[validation_start:train_count])
    held_truth = initial[validation_start:train_count]
    model = fit_additive_program_model(
        codes[:train_count], initial[:train_count], option_sizes)
    fit_seconds = time.perf_counter() - fit_start
    start = time.perf_counter()
    predicted = model.predict(codes[uniform_count:])
    # Tie by the original seeded candidate index for reproducibility.
    ordering = np.lexsort((np.arange(uniform_count, len(programs)), -predicted))
    finalists = ordering[:finalist_count] + uniform_count
    screening_seconds = time.perf_counter() - start
    start = time.perf_counter()
    finalist_scores = selection_scores(finalists)
    finalist_seconds = time.perf_counter() - start
    baseline_index = int(np.argmax(initial))
    best_finalist = int(np.argmax(finalist_scores))
    seed_best = int(np.argmax(initial[:train_count]))
    surrogate_index = (seed_best if initial[seed_best] >=
                       finalist_scores[best_finalist] else
                       int(finalists[best_finalist]))
    selected = {"uniform_exact": baseline_index,
                "additive_screened": surrogate_index}
    fold = prepare_supported_fold(
        day1, (day2[0][support_ids], day2[1][support_ids]),
        (day2[0][query_ids], day2[1][query_ids]),
        features_per_site=2, support_weight=4.0)
    query = score_programs([fold], programs[list(selected.values())],
                           features_per_site=2, backend=backend)
    lookup = {int(index): float(score) for index, score in
              zip(finalists, finalist_scores)}
    return {
        "sources": [meta1["source_record_sha256"],
                    meta2["source_record_sha256"]],
        "support_indices": support_ids.tolist(),
        "query_indices": query_ids.tolist(),
        "surrogate_heldout_spearman": float(spearmanr(
            held_prediction, held_truth).statistic),
        "surrogate_heldout_mae": float(np.mean(np.abs(
            held_prediction - held_truth))),
        "times_seconds": {"shared_seed_exact_5000": seed_seconds,
                          "uniform_extra_exact_5000": baseline_extra_seconds,
                          "additive_fit_with_heldout_diagnostic": fit_seconds,
                          "surrogate_screen_remaining": screening_seconds,
                          "finalist_exact": finalist_seconds},
        "selection": {name: {
            "candidate_index": index,
            "program": programs[index].tolist(),
            "physical_contacts": physical_contacts(
                programs[index], meta1["pairs"]),
            "selection_score": (float(initial[index])
                                if index < uniform_count else lookup[index]),
            "query_balanced_accuracy": float(query[i])}
            for i, (name, index) in enumerate(selected.items())},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mono-cache", type=Path, required=True)
    parser.add_argument("--pair-cache", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--candidates", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=int, default=5000)
    parser.add_argument("--finalists", type=int, default=5000)
    parser.add_argument("--backend", choices=("numpy", "rust"), default="rust")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.output.exists() or args.train < 5 or args.finalists < 1 or
            args.candidates <= args.train + args.finalists):
        parser.error("new output and valid disjoint candidate budgets required")
    first_meta = load_mixed(args.mono_cache, args.pair_cache,
                            args.subjects[0], 1)[1]
    options = edge_mode_options(first_meta)
    start = time.perf_counter()
    programs, codes = uniform_program_prefix_with_codes(
        options, args.candidates, args.seed)
    sampling_seconds = time.perf_counter() - start
    topology = (first_meta["site_names"], first_meta["pairs"])
    rows = {}
    for subject in args.subjects:
        rows[subject] = analyze_subject(
            args.mono_cache, args.pair_cache, subject, programs, codes,
            topology, tuple(len(group) for group in options),
            train_count=args.train, finalist_count=args.finalists,
            backend=args.backend)
        print(subject, {name: round(value["query_balanced_accuracy"], 3)
                        for name, value in rows[subject]["selection"].items()},
              flush=True)
    result = {
        "schema": "openalterego.hyser-additive-surrogate-search.v2",
        "scope": "Retrospective exposed forearm subjects; target query labels excluded from screening and program selection",
        "subjects": args.subjects, "candidate_count": args.candidates,
        "candidate_seed": args.seed, "candidate_pool_sha256": hashlib.sha256(
            programs.tobytes()).hexdigest(),
        "train_exact": args.train, "finalists_exact": args.finalists,
        "baseline_exact": args.train + args.finalists,
        "surrogate": "ridge-10 additive one-hot effects of four complete edge/mode grammar choices; inner heldout last 20 percent of seed scores",
        "backend": args.backend,
        "shared_sampling_seconds": sampling_seconds,
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
