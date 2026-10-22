"""Paired utterance bootstrap for the frozen Gaddy trial-range result."""

import argparse
import json
from pathlib import Path

import numpy as np


def bootstrap_session(session: dict, draws: int, rng) -> dict[str, np.ndarray]:
    counts = np.asarray(session["evaluation_frame_counts"], dtype=np.float64)
    if len(counts) == 0 or np.any(counts <= 0):
        raise ValueError("missing positive utterance frame counts")
    indices = rng.integers(0, len(counts), size=(draws, len(counts)))
    weights = counts[indices]
    return {name: (np.asarray(result["utterance_mae"])[indices] *
                   weights).sum(axis=1) / weights.sum(axis=1)
            for name, result in session["results"].items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    record = json.loads(args.input.read_text(encoding="utf-8"))
    if (record.get("schema") != "openalterego.gaddy-locked-trial-range.v1" or
            set(record["sessions"]) != {"5-4", "5-8"}):
        parser.error("unexpected frozen result")
    rng = np.random.default_rng(42)
    draws = 10000
    samples = {name: bootstrap_session(session, draws, rng)
               for name, session in record["sessions"].items()}
    comparisons = {}
    for baseline in ("index_fixed_six", "earlier_session_six", "all_eight"):
        differences = {session: row["multisession_six"] - row[baseline]
                       for session, row in samples.items()}
        differences["two_session_mean"] = np.mean(
            list(differences.values()), axis=0)
        comparisons[baseline] = {
            session: {"selected_minus_baseline_ci95":
                      [float(value) for value in np.quantile(sample,
                                                               [0.025, 0.975])]}
            for session, sample in differences.items()}
    result = {"schema": "openalterego.gaddy-locked-trial-range-audit.v1",
              "scope": "Descriptive paired utterance resampling within two exposed sessions; not a new-person uncertainty interval",
              "seed": 42, "draws": draws, "comparisons": comparisons}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
