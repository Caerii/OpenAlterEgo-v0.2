"""Post-gate, transductive Hyser normalization diagnostic; not a live decoder."""

import argparse
import json
from pathlib import Path

import numpy as np

from openalterego.research.hyser_search import ridge_test
from scripts.hyser_program_search import load_session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    result = {}
    for subject in selection["subjects"]:
        train, _ = load_session(args.cache_dir, subject, 1)
        test, _ = load_session(args.cache_dir, subject, 2)
        source, target = train[0], test[0]
        transformed = {
            "raw": target,
            "target_mean_center": target - target.mean(axis=0) + source.mean(axis=0),
            "target_mean_sd_align": (
                (target - target.mean(axis=0)) *
                np.maximum(source.std(axis=0), 1e-6) /
                np.maximum(target.std(axis=0), 1e-6) + source.mean(axis=0)),
        }
        result[subject] = {}
        for name in ("coverage_greedy", "fixed_grid"):
            channels = selection["choices"][subject]["strategies"][name]["channels"]
            result[subject][name] = {
                mode: ridge_test(train, (features, test[1]), channels)
                for mode, features in transformed.items()}
    print(json.dumps({"scope": "post-hoc full-target-session transductive diagnostic",
                      "subjects": result}, indent=2))


if __name__ == "__main__":
    main()
