"""Torchhd random-projection centroid control on fixed SilentWear channels."""

import argparse
from importlib.metadata import version
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torchhd
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from torchhd.embeddings import Projection
from torchhd.models import Centroid

from openalterego.research.program_search import normalize_batches
from scripts.silentwear_program_search import subject_sessions


SPREAD = (0, 2, 5, 8, 11, 13)
COLUMNS = [column for channel in SPREAD for column in (2 * channel, 2 * channel + 1)]


def session_features(batches):
    features, labels = normalize_batches(batches, "recording_robust")
    return features[:, COLUMNS], labels.astype(np.int64)


def fit_and_score(train, test_loader, *, dimensions: int, seed: int):
    train_x, train_y = train
    torch.manual_seed(seed)
    started = time.perf_counter()
    scaler = StandardScaler().fit(train_x)
    scaled_train = torch.from_numpy(scaler.transform(train_x).astype(np.float32))
    encoder = Projection(len(COLUMNS), dimensions)
    classifier = Centroid(dimensions, 9)
    with torch.inference_mode():
        classifier.add(torchhd.normalize(encoder(scaled_train)),
                       torch.from_numpy(train_y))
    fit_seconds = time.perf_counter() - started
    test_x, test_y = test_loader()
    started = time.perf_counter()
    scaled_test = torch.from_numpy(scaler.transform(test_x).astype(np.float32))
    with torch.inference_mode():
        predicted = classifier(torchhd.normalize(encoder(scaled_test))).argmax(1).numpy()
    inference_seconds = time.perf_counter() - started
    return {"test_balanced_accuracy": float(balanced_accuracy_score(test_y, predicted)),
            "fit_wall_seconds": fit_seconds,
            "inference_wall_seconds": inference_seconds,
            "model_bytes_float32": int((len(COLUMNS) + 9) * dimensions * 4),
            "event_counts": {"train": len(train_y), "test": len(test_y)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", default=["S01", "S02", "S03"])
    parser.add_argument("--dimensions", nargs="+", type=int, default=[512, 2048, 8192])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (not set(args.subjects).issubset({"S01", "S02", "S03"}) or
            any(dim < 1 for dim in args.dimensions) or args.threads < 1):
        parser.error("invalid development subjects, dimensions, or thread count")
    torch.set_num_threads(args.threads)
    config = {"subjects": args.subjects, "dimensions": args.dimensions,
              "seed": args.seed, "threads": args.threads,
              "channels": list(SPREAD), "normalization": "recording_robust",
              "feature_scaler": "train-only StandardScaler", "embedding": "Torchhd Projection MAP + normalize",
              "classifier": "Torchhd Centroid cosine"}
    if args.output.exists():
        if not args.resume:
            parser.error("output exists; use --resume")
        result = json.loads(args.output.read_text(encoding="utf-8"))
        if result.get("schema") != "openalterego.torchhd-fixed-layout.v1" or result.get("config") != config:
            parser.error("existing run does not match configuration")
    else:
        result = {"schema": "openalterego.torchhd-fixed-layout.v1",
                  "config": config, "torch_version": torch.__version__,
                  "torchhd_version": version("torch-hd"), "subjects": {},
                  "scope": "Retrospective fixed recorded channels; session 3 is previously explored, not a fresh holdout."}
    for subject in args.subjects:
        if subject in result["subjects"]:
            continue
        load_session, provenance, manifest_hash = subject_sessions(args.root, subject, args.cache_dir)
        train = session_features(load_session(1) + load_session(2))
        rows = {}
        for dimensions in args.dimensions:
            row = fit_and_score(train, lambda: session_features(load_session(3)),
                                dimensions=dimensions, seed=args.seed)
            rows[str(dimensions)] = row
            print(subject, dimensions, row["test_balanced_accuracy"],
                  row["fit_wall_seconds"], flush=True)
        result["subjects"][subject] = {"dimensions": rows,
                                       "input_manifest_sha256": manifest_hash,
                                       "inputs": provenance}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f"{args.output.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, args.output)


if __name__ == "__main__":
    main()
