"""Score only the four programs frozen in research protocol 66."""

import argparse
import json
from pathlib import Path

import numpy as np

from openalterego.research.dense_regression import (
    predict_regression_programs, prepare_regression_fold)
from scripts.gaddy_channel_search import (
    design, load, manifest_identity, normalize_rows)


PROGRAMS = {
    "multisession_six": (0, 1, 2, 3, 4, 6),
    "earlier_session_six": (0, 1, 3, 4, 6, 7),
    "index_fixed_six": (0, 1, 2, 4, 5, 7),
    "all_eight": tuple(range(8)),
}
EXPECTED_QUERY = {
    "parallel-5-4-ids100-199": "emg_data/voiced_parallel_data/5-4/",
    "parallel-5-8-ids100-199": "emg_data/voiced_parallel_data/5-8/",
}


def valid_ids(path: Path) -> list[int]:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("start") != 100 or manifest.get("count") != 100:
        raise ValueError("query manifest does not describe locked ID range")
    ids = sorted({int(name.split("_", 1)[0]) for name in manifest["files"]})
    if ids != list(range(100, 200)):
        raise ValueError("query manifest has incomplete trial range")
    return [trial for trial in ids if json.loads(
        (path / f"{trial}_info.json").read_text(encoding="utf-8"))[
            "sentence_index"] >= 0]


def score_session(train, query_path: Path, cache_dir: Path) -> dict:
    identity = manifest_identity(query_path)
    if EXPECTED_QUERY.get(query_path.name) != identity["source_path"]:
        raise ValueError("query session differs from frozen protocol")
    ids = valid_ids(query_path)
    rows = load(query_path, cache_dir)
    if len(rows) != len(ids) or len(rows) <= 10:
        raise ValueError("query feature rows do not match valid trial IDs")
    evaluation = design(normalize_rows(rows[10:], rows[:10]))
    fold = prepare_regression_fold(train, evaluation, features_per_channel=8)
    offsets = np.r_[0, np.cumsum([len(row[1]) for row in rows[10:]])]
    results = {}
    for name, channels in PROGRAMS.items():
        pred = predict_regression_programs(
            fold, np.asarray([channels], dtype=np.int16))[:, 0]
        error = np.abs(pred - evaluation[1])
        results[name] = {
            "frame_mae": float(error.mean()),
            "utterance_mae": [float(error[offsets[i]:offsets[i+1]].mean())
                              for i in range(len(offsets)-1)],
        }
    return {"source": identity, "calibration_trial_ids": ids[:10],
            "evaluation_trial_ids": ids[10:],
            "evaluation_frame_counts": [len(row[1]) for row in rows[10:]],
            "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--query-5-4", type=Path, required=True)
    parser.add_argument("--query-5-8", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    train_identity = manifest_identity(args.train)
    if (args.train.name != "parallel-5-5-first100" or
            train_identity["source_path"] !=
            "emg_data/voiced_parallel_data/5-5/"):
        parser.error("training source differs from locked protocol")
    train_manifest = json.loads((args.train / "manifest.json").read_text())
    if sorted({int(name.split("_", 1)[0]) for name in
               train_manifest["files"]}) != list(range(100)):
        parser.error("training manifest lacks IDs 0-99")
    train = design(normalize_rows(load(args.train, args.cache_dir)))
    sessions = {}
    for name, path in (("5-4", args.query_5_4), ("5-8", args.query_5_8)):
        if path.name != f"parallel-{name}-ids100-199":
            parser.error("query path does not match locked session")
        sessions[name] = score_session(train, path, args.cache_dir)
        if sessions[name]["source"]["source_md5"] != train_identity["source_md5"]:
            raise ValueError("query source archive differs from training source")
    mean_mae = {name: float(np.mean([
        session["results"][name]["frame_mae"]
        for session in sessions.values()])) for name in PROGRAMS}
    result = {"schema": "openalterego.gaddy-locked-trial-range.v1",
              "protocol_commit": "19a1661", "scope":
              "Previously unscored trial IDs in exposed one-speaker sessions; vocalized acoustic energy",
              "train": train_identity, "programs": PROGRAMS,
              "sessions": sessions, "mean_session_frame_mae": mean_mae}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(mean_mae, indent=2))


if __name__ == "__main__":
    main()
