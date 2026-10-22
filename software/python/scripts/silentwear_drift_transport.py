"""Ask whether measured day-one/two channel drift forecasts day three."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from openalterego.research.channel_stability import class_contrast
from openalterego.research.program_search import normalize_batches
from scripts.silentwear_program_search import subject_sessions


def _contrast(batches, normalization):
    return class_contrast(*normalize_batches(batches, normalization))


def analyze_subject(root: Path, cache_dir: Path, subject: str) -> dict:
    load_session, provenance, manifest_hash = subject_sessions(
        root, subject, cache_dir)
    days = [load_session(index) for index in (1, 2, 3)]
    modes = {}
    for norm in ("none", "recording_robust"):
        early = [_contrast(day, norm) for day in days]
        (c1, w1), (c2, w2), (c3, _) = early
        scale = np.maximum((w1 + w2) / 2, .01)
        early_shift = np.mean((c1 - c2) ** 2 / scale, axis=(0, 2))
        future_shift = np.mean(((c1 + c2) / 2 - c3) ** 2 / scale,
                               axis=(0, 2))
        early_signal = np.mean(((c1 + c2) / 2) ** 2 / scale,
                               axis=(0, 2))
        future_signal = np.mean(c3 ** 2 / scale, axis=(0, 2))
        first_half = [_contrast(day[:2], norm)[0] for day in days[:2]]
        second_half = [_contrast(day[2:], norm)[0] for day in days[:2]]
        early_batch_shift = np.mean((first_half[0] - first_half[1]) ** 2,
                                    axis=(0, 2))
        late_batch_shift = np.mean((second_half[0] - second_half[1]) ** 2,
                                   axis=(0, 2))
        modes[norm] = {
            "channel_count": len(early_shift),
            "rank_correlations": {
                "same_day_pair_batch_replication": float(spearmanr(
                    early_batch_shift, late_batch_shift).statistic),
                "early_drift_to_day3_drift": float(spearmanr(
                    early_shift, future_shift).statistic),
                "early_signal_to_day3_signal": float(spearmanr(
                    early_signal, future_signal).statistic),
                "early_drift_to_day3_signal": float(spearmanr(
                    early_shift, future_signal).statistic),
            },
            "channel_metrics": {
                "early_batch_shift": early_batch_shift.tolist(),
                "late_batch_shift": late_batch_shift.tolist(),
                "early_drift": early_shift.tolist(),
                "day3_drift": future_shift.tolist(),
                "early_signal": early_signal.tolist(),
                "day3_signal": future_signal.tolist(),
            },
        }
    return {"manifest_sha256": manifest_hash,
            "source_batches": provenance, "normalizations": modes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", default=["S01", "S02", "S03"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.output.exists() or
            not set(args.subjects).issubset({"S01", "S02", "S03"})):
        parser.error("new output and development subjects required")
    rows = {}
    for subject in args.subjects:
        rows[subject] = analyze_subject(args.root, args.cache_dir, subject)
        print(subject, {norm: data["rank_correlations"] for norm, data in
                        rows[subject]["normalizations"].items()}, flush=True)
    result = {
        "schema": "openalterego.silentwear-channel-drift-transport.v1",
        "scope": "Post hoc exposed silent-command sessions and recorded channels; descriptive ranks over fourteen dependent channels, not physical reseating effects",
        "split": "days 1 and 2 produce early signal and drift; day 3 is diagnostic outcome; first two versus last three batches of days 1/2 assess within-pair reproducibility",
        "metric": "class-balanced centroid contrasts after removal of channelwise mean over classes; early within-class variance floor 0.01 for cross-day measures",
        "subjects": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
