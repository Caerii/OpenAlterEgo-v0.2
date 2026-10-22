"""Replay frozen facial channel choices with four-bin temporal features."""

import argparse
import json
from pathlib import Path

import numpy as np

from openalterego.research.dense_programs import prepare_fold, score_programs
from scripts.faceemg11_program_search import split_subject


NAMES = ("personal_train_cv", "personal_forward", "global_train_cv",
         "global_forward", "fixed_index_spread")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--coarse-score", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    coarse = json.loads(args.coarse_score.read_text(encoding="utf-8"))
    if (selection.get("schema") != "openalterego.faceemg11-method-selection.v1" or
            coarse.get("schema") != "openalterego.faceemg11-method-score.v1"):
        raise ValueError("unexpected selection or score")
    result = {
        "schema": "openalterego.faceemg11-four-bin-replay.v1",
        "scope": "Post hoc representation comparison on previously exposed query blocks",
        "coarse_selection_sha256": coarse["selection_sha256"],
        "feature_count_per_channel": 8,
        "alpha": 10.0, "subjects": {},
    }
    for subject in selection["method_check_subjects"]:
        stem = f"sub-{subject}_trials-temporal"
        metadata = json.loads((args.feature_cache / (stem + ".json")).read_text(
            encoding="utf-8"))
        if metadata.get("schema") != "openalterego.faceemg11-four-bin.v1":
            raise ValueError(f"unexpected temporal source: {subject}")
        with np.load(args.feature_cache / (stem + ".npz"), allow_pickle=False) as data:
            features = data["features"].astype(np.float64)
            labels = data["labels"].copy()
            groups = data["groups"].copy()
        if (features.shape != (330, 160) or
                metadata["source_member_sha256"] !=
                selection["method_choices"][subject]["source_member_sha256"]):
            raise ValueError(f"source or feature shape differs: {subject}")
        train, validation, query = split_subject(features, labels, groups)
        fit = (np.vstack((train[0], validation[0])),
               np.r_[train[1], validation[1]])
        fold = prepare_fold(fit, query, features_per_site=8)
        programs = selection["method_choices"][subject]["programs"]
        designs = np.asarray([programs[name] for name in NAMES], dtype=np.int16)
        values = score_programs([fold], designs, features_per_site=8)
        full = score_programs([fold], np.arange(20, dtype=np.int16)[None, :],
                              features_per_site=8)[0]
        scores = {name: float(value) for name, value in zip(NAMES, values)}
        scores["all_twenty_channel_reference"] = float(full)
        coarse_scores = coarse["subjects"][subject]["query_balanced_accuracy"]
        result["subjects"][subject] = {
            "source_member_sha256": metadata["source_member_sha256"],
            "temporal_query_balanced_accuracy": scores,
            "coarse_query_balanced_accuracy": coarse_scores,
            "personal_cv_temporal_minus_coarse": (
                scores["personal_train_cv"] - coarse_scores["personal_train_cv"]),
        }
        print(subject, {name: round(value, 3) for name, value in scores.items()},
              flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
