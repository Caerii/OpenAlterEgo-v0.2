"""Evaluate fixed searched programs with identical labeled support/query trials."""

import argparse
import json
from pathlib import Path

import numpy as np

from openalterego.research.dense_programs import score_programs
from openalterego.research.hyser_fewshot import support_query
from openalterego.research.supported_programs import prepare_supported_fold
from scripts.hyser_mixed_program_pilot import load_mixed


PROGRAMS = ("beam", "random", "fixed_mixed", "fixed_monopolar",
            "beam_same_contacts_monopolar")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mono-cache", type=Path, required=True)
    parser.add_argument("--pair-cache", type=Path, required=True)
    parser.add_argument("--selection-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    selection = json.loads(args.selection_result.read_text(encoding="utf-8"))
    if selection.get("schema") != "openalterego.hyser-compositional-search.v1":
        raise ValueError("unexpected program selection artifact")
    result = {"schema": "openalterego.hyser-compositional-calibration.v1",
              "scope": "Retrospective forearm gesture replay; support/query disjoint but class-block order is not globally chronological",
              "shots_per_class": 1, "support_weights": (0.0, 1.0, 4.0),
              "alpha": 10.0, "subjects": {}}
    for subject, earlier in selection["subjects"].items():
        day1, meta1 = load_mixed(args.mono_cache, args.pair_cache, subject, 1)
        day2, meta2 = load_mixed(args.mono_cache, args.pair_cache, subject, 2)
        if (meta1["source_record_sha256"] != earlier["day1_source_record_sha256"] or
                meta2["source_record_sha256"] != earlier["day2_source_record_sha256"]):
            raise ValueError(f"source identity changed: {subject}")
        support, query = support_query(day2[1], 1)
        designs = np.asarray([earlier["choices"][name]["channels"]
                              for name in PROGRAMS], dtype=np.int16)
        scores = {}
        for weight in result["support_weights"]:
            fold = prepare_supported_fold(
                day1, (day2[0][support], day2[1][support]),
                (day2[0][query], day2[1][query]),
                features_per_site=2, support_weight=weight)
            score = score_programs([fold], designs, features_per_site=2)
            scores[str(weight)] = {name: float(value)
                                   for name, value in zip(PROGRAMS, score)}
        result["subjects"][subject] = {
            "day1_source_record_sha256": meta1["source_record_sha256"],
            "day2_source_record_sha256": meta2["source_record_sha256"],
            "support_trials": len(support), "query_trials": len(query),
            "query_balanced_accuracy": scores}
        print(subject, {name: round(scores["1.0"][name], 3)
                        for name in PROGRAMS}, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
