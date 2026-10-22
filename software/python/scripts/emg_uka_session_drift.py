"""Measure how well small support sets predict EMG feature drift.

This is a retrospective feature-prediction assay on repeated audible prompts.
Session templates use only earlier numbered sessions. Within a target session,
support uses lower utterance IDs; the corpus does not verify that those IDs
encode acquisition order, so this is not a prospective online calibration test.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.emg_uka_session_program import check_source, session


SESSIONS = tuple(f"{index:03d}" for index in range(1, 9))
PREFIXES = (2, 4, 6)
MODELS = ("none", "global", "channel", "channel_feature",
          "full_feature", "unlabeled_global", "unlabeled_channel")


def recording_ids(root: Path, session_id: str) -> np.ndarray:
    subset = (root / "Subsets" / "test.audible").read_text(
        encoding="utf-8")
    key = f"emg_008-{session_id}:"
    rows = [line for line in subset.splitlines() if line.startswith(key)]
    if len(rows) != 1:
        raise ValueError("missing session subset")
    identifiers = rows[0].split(":", 1)[1].split()
    numbers = [int(identifier.rsplit("-", 1)[1]) for identifier in
               identifiers]
    if len(numbers) != 10 or len(set(numbers)) != 10:
        raise ValueError("unexpected session utterance IDs")
    return np.asarray(numbers, dtype=np.int32)


def estimate_shift(residual: np.ndarray, model: str) -> np.ndarray:
    """Broadcast a calibration residual mean under a declared complexity."""
    if residual.ndim != 2 or residual.shape[1] != 60 or not len(residual):
        raise ValueError("expected calibration rows with 60 EMG features")
    blocks = residual.reshape(len(residual), 6, 5, 2)
    if model == "none":
        shift = np.zeros((6, 5, 2))
    elif model == "global":
        shift = np.full((6, 5, 2), blocks.mean())
    elif model == "channel":
        shift = np.broadcast_to(blocks.mean(axis=(0, 2, 3))[:, None, None],
                                (6, 5, 2))
    elif model == "channel_feature":
        shift = np.broadcast_to(blocks.mean(axis=(0, 2))[:, None, :],
                                (6, 5, 2))
    elif model == "full_feature":
        shift = blocks.mean(axis=0)
    else:
        raise ValueError("unknown drift model")
    return shift.reshape(60)


def assay(root: Path, manifest: dict) -> dict:
    loaded = {name: session(root, name) for name in SESSIONS}
    labels = tuple(sorted(loaded["001"][1]))
    if any(set(prompts) != set(labels) for _, prompts, _ in loaded.values()):
        raise ValueError("speaker-008 repeated prompt sets differ")
    results = []
    for target_index in range(1, len(SESSIONS)):
        target = SESSIONS[target_index]
        past = SESSIONS[:target_index]
        prompt_index = {name: {prompt: index for index, prompt in
                               enumerate(loaded[name][1])}
                        for name in past}
        templates = {prompt: np.mean([
            loaded[name][0][prompt_index[name][prompt]] for name in past],
            axis=0) for prompt in labels}
        population_template = np.mean(list(templates.values()), axis=0)
        target_x, target_prompts, _ = loaded[target]
        identifiers = recording_ids(root, target)
        order = np.argsort(identifiers)
        residual = target_x - np.asarray([templates[prompt] for prompt in
                                           target_prompts])
        for prefix in PREFIXES:
            calibration = residual[order[:prefix]]
            future = residual[order[prefix:]]
            model_mae = {model: float(np.mean(abs(
                future - estimate_shift(calibration, model))))
                for model in MODELS[:5]}
            unlabeled_calibration = target_x[order[:prefix]] - population_template
            for model, base in (("unlabeled_global", "global"),
                                ("unlabeled_channel", "channel")):
                model_mae[model] = float(np.mean(abs(
                    future - estimate_shift(unlabeled_calibration, base))))
            results.append({
                "target_session": target,
                "template_sessions": past,
                "prefix_utterance_ids": identifiers[order[:prefix]].tolist(),
                "calibration_count": prefix,
                "evaluation_count": len(future),
                "mae_log_adc_feature": model_mae,
            })
    summary = {}
    for prefix in PREFIXES:
        rows = [row for row in results if row["calibration_count"] == prefix]
        sampled = np.random.default_rng(42 + prefix).integers(
            0, len(rows), size=(10000, len(rows)))
        summary[str(prefix)] = {
            model: {"mean_session_mae": float(np.mean([
                row["mae_log_adc_feature"][model] for row in rows])),
                "improved_sessions_vs_none": int(sum(
                    row["mae_log_adc_feature"][model] <
                    row["mae_log_adc_feature"]["none"] for row in rows)),
                "paired_none_minus_model_mean": float(np.mean([
                    row["mae_log_adc_feature"]["none"] -
                    row["mae_log_adc_feature"][model] for row in rows])),
                "paired_session_bootstrap_ci95": [float(value) for value in
                    np.quantile(np.asarray([
                        row["mae_log_adc_feature"]["none"] -
                        row["mae_log_adc_feature"][model] for row in rows
                    ])[sampled].mean(axis=1), [0.025, 0.975])],
                "bootstrap_draws": 10000}
            for model in MODELS}
    return {
        "schema": "openalterego.emg-uka-session-drift.v1",
        "scope": "Retrospective speaker-008 repeated audible prompts; feature prediction under earlier-session templates and target-session lower-ID support; utterance-ID time order unverified, no online calibration or identified contact impedance or physical electrode intervention",
        "source_download": manifest["download"],
        "archive_sha256": manifest["downloaded_archive_sha256"],
        "session_content_digests": {name: item[2] for name, item in
                                    loaded.items()},
        "feature": "per-utterance centered ADC; five relative-time bins; log1p RMS and mean absolute adjacent difference per channel",
        "models": MODELS,
        "prefixes": PREFIXES,
        "results": results,
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    manifest = check_source(args.root, args.manifest)
    result = assay(args.root, manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
