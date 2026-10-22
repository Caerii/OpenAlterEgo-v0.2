"""Measure compositional-search compute scaling, not biological accuracy."""

import argparse
import json
import math
import platform
import time
from pathlib import Path

import numpy as np

from openalterego.research.compositional_search import beam_search
from openalterego.research.csl_protocol import (
    CONTACT_PAIRS, feasible_recorded_channels,
    minimum_contacts_to_complete)
from openalterego.research.dense_programs import prepare_cv_folds, score_programs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channels", nargs="+", type=int, default=[20, 40, 80])
    parser.add_argument("--beam-size", type=int, default=4)
    parser.add_argument("--trials", type=int, default=165)
    parser.add_argument("--features-per-site", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--csl-contact-budget", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    if (args.beam_size < 1 or args.repeats < 1 or args.trials < 33 or
            args.features_per_site < 1 or
            any(channels < 6 or channels > 1024 for channels in args.channels)):
        parser.error("invalid benchmark dimensions")
    if (args.csl_contact_budget is not None and
            (args.channels != [len(CONTACT_PAIRS)] or
             args.csl_contact_budget < 7)):
        parser.error("CSL contact budget requires 38 channels and at least 7 contacts")
    rng = np.random.default_rng(20261022)
    labels = np.arange(args.trials) % 11
    result = {
        "schema": "openalterego.dense-search-compute-scaling.v1",
        "scope": "Gaussian synthetic matrices; compute scaling only; scores have no physiological meaning",
        "beam_size": args.beam_size,
        "trials": args.trials,
        "classes": 11,
        "features_per_site": args.features_per_site,
        "repeats": args.repeats,
        "csl_contact_budget": args.csl_contact_budget,
        "machine": platform.platform(),
        "numpy": np.__version__,
        "rows": [],
    }
    for channels in args.channels:
        features = rng.standard_normal((args.trials,
                                        channels * args.features_per_site))
        folds = prepare_cv_folds(
            features, labels, features_per_site=args.features_per_site)
        options = tuple(tuple((channel,) for channel in range(channels))
                        for _ in range(6))
        score = lambda programs: score_programs(
            folds, programs, features_per_site=args.features_per_site)
        feasible = (None if args.csl_contact_budget is None else
                    lambda program: feasible_recorded_channels(
                        program, max_contacts=args.csl_contact_budget,
                        max_afe_channels=6))
        can_complete = (None if args.csl_contact_budget is None else
                        lambda program, remaining:
                        minimum_contacts_to_complete(
                            program, len(program) + remaining)
                        <= args.csl_contact_budget)
        # Warm the solver before recording timings on each matrix size.
        beam_search(options, score, beam_size=args.beam_size,
                    feasible=feasible, can_complete=can_complete)
        timings = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            selected = beam_search(options, score, beam_size=args.beam_size,
                                   feasible=feasible,
                                   can_complete=can_complete)
            timings.append(time.perf_counter() - started)
        row = {
            "channels": channels,
            "full_six_channel_space": math.comb(channels, 6),
            "program_evaluations": selected.evaluations,
            "stage_evaluations": selected.stage_evaluations,
            "search_seconds_repetitions": timings,
            "search_seconds_median": float(np.median(timings)),
        }
        if feasible is not None:
            row["selected_channels"] = selected.channels
            row["selected_contacts"] = sorted({
                contact for channel in selected.channels
                for contact in CONTACT_PAIRS[channel]})
        result["rows"].append(row)
        print(row, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
