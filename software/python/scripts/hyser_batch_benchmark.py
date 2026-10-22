"""Benchmark batched versus per-design next-day ridge scoring on one Hyser pair."""

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np

from openalterego.research.hyser_features import site_coordinates
from openalterego.research.hyser_search import ridge_test
from openalterego.research.hyser_spatial_search import candidate_designs
from openalterego.research.hyser_transfer import prepare_day_pair, score_day_pair
from scripts.hyser_program_search import load_session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--subject", default="09")
    parser.add_argument("--designs", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    train, meta = load_session(args.cache_dir, args.subject, 1)
    target, target_meta = load_session(args.cache_dir, args.subject, 2)
    sites = site_coordinates(tuple(meta["site_names"]))
    designs = candidate_designs(sites, count=args.designs)
    pair = prepare_day_pair(train, target)
    batched_times, scalar_times = [], []
    batch = None
    scalar = None
    for _ in range(3):
        start = time.perf_counter()
        batch = score_day_pair(pair, designs)
        batched_times.append(time.perf_counter() - start)
        start = time.perf_counter()
        scalar = np.asarray([ridge_test(train, target, list(map(int, row)))
                             for row in designs])
        scalar_times.append(time.perf_counter() - start)
    maximum_error = float(np.max(np.abs(batch - scalar)))
    if maximum_error > 1e-12:
        raise ValueError(f"batched/scalar score mismatch: {maximum_error}")
    result = {"schema": "openalterego.hyser-batch-benchmark.v1",
              "scope": "Single-host warm-cache scoring only; excludes acquisition, feature extraction and training-data preparation",
              "subject": args.subject, "designs": len(designs), "repeats": 3,
              "train_source_record_sha256": meta["source_record_sha256"],
              "test_source_record_sha256": target_meta["source_record_sha256"],
              "machine": platform.platform(), "python": platform.python_version(),
              "numpy": np.__version__,
              "batched_seconds": batched_times,
              "scalar_seconds": scalar_times,
              "batched_median_seconds": statistics.median(batched_times),
              "scalar_median_seconds": statistics.median(scalar_times),
              "median_speedup": statistics.median(scalar_times) / statistics.median(batched_times),
              "max_abs_score_error": maximum_error}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in (
        "batched_median_seconds", "scalar_median_seconds", "median_speedup",
        "max_abs_score_error")}, indent=2))


if __name__ == "__main__":
    main()
