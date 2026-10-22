"""Two-day class-signal channel prior with a fixed six-channel comparator."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from openalterego.research.channel_stability import class_contrast
from openalterego.research.dense_programs import prepare_fold, score_programs_by_fold
from openalterego.research.program_search import normalize_batches
from openalterego.research.virtual_montage import fixed_subsets
from scripts.silentwear_additive_screen import _freeze
from scripts.silentwear_program_search import subject_sessions


def _query_accuracy(first, second, third, channels, normalization):
    fit_x, fit_y = normalize_batches(first + second, normalization)
    query_x, query_y = normalize_batches(third, normalization)
    columns = [column for channel in channels
               for column in (2 * channel, 2 * channel + 1)]
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, max_iter=2000, class_weight="balanced", random_state=42))
    model.fit(fit_x[:, columns], fit_y)
    return float(balanced_accuracy_score(
        query_y, model.predict(query_x[:, columns])))


def analyze(root: Path, cache_dir: Path, subject: str,
            selection_path: Path) -> dict:
    load_session, provenance, manifest_hash = subject_sessions(
        root, subject, cache_dir)
    first, second = load_session(1), load_session(2)
    candidates = []
    for normalization in ("none", "recording_robust"):
        day1 = normalize_batches(first, normalization)
        day2 = normalize_batches(second, normalization)
        contrast1, within1 = class_contrast(*day1)
        contrast2, within2 = class_contrast(*day2)
        scale = np.maximum((within1 + within2) / 2, .01)
        signal = np.mean(((contrast1 + contrast2) / 2) ** 2 / scale,
                         axis=(0, 2))
        channels = sorted(map(int, np.lexsort(
            (np.arange(14), -signal))[:6]))
        folds = (prepare_fold(day1, day2, features_per_site=2),
                 prepare_fold(day2, day1, features_per_site=2))
        directional = score_programs_by_fold(
            folds, np.asarray([channels], dtype=np.int16),
            features_per_site=2, backend="rust")[:, 0]
        candidates.append({"channels": channels,
                           "normalization": normalization,
                           "early_signal_by_channel": signal.tolist(),
                           "directional_ridge_accuracy": directional.tolist(),
                           "selection_score": float(directional.min())})
    chosen = max(candidates, key=lambda row: row["selection_score"])
    fixed = {"channels": fixed_subsets()["spread"],
             "normalization": "recording_robust"}
    receipt = _freeze(selection_path, {
        "schema": "openalterego.silentwear-signal-prior-selection.v1",
        "subject": subject, "manifest_sha256": manifest_hash,
        "candidate_modes": candidates,
        "choice": {"channels": chosen["channels"],
                   "normalization": chosen["normalization"]},
        "fixed": fixed,
        "scope": "Local prequery record; retrospective exposed sessions"})
    third = load_session(3)
    return {"manifest_sha256": manifest_hash,
            "source_batches": provenance,
            "selection_record": str(selection_path),
            "selection_receipt_sha256": receipt,
            "candidate_modes": candidates,
            "choice": {"channels": chosen["channels"],
                       "normalization": chosen["normalization"],
                       "query_balanced_accuracy": _query_accuracy(
                           first, second, third, chosen["channels"],
                           chosen["normalization"])},
            "fixed_spread_robust": {**fixed,
                "query_balanced_accuracy": _query_accuracy(
                    first, second, third, fixed["channels"],
                    fixed["normalization"])}}


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
        selection_path = args.output.with_name(
            f"{args.output.stem}-{subject}-selection.json")
        rows[subject] = analyze(args.root, args.cache_dir,
                                subject, selection_path)
        print(subject, rows[subject]["choice"], flush=True)
    result = {
        "schema": "openalterego.silentwear-signal-prior.v1",
        "scope": "Post hoc method development on exposed silent-command sessions; virtual recorded-channel subsets only",
        "selection": "top six channels by mean day1/day2 class-contrast energy divided by early within-class variance; choose raw versus recording-robust with two-direction minimum ridge score",
        "final_decoder": "balanced C=1 logistic fit on sessions 1+2, evaluated on session 3",
        "subjects": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
