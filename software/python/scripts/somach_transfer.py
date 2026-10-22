"""Compare two public electrode configurations without a physical experiment."""

import argparse
import json
from pathlib import Path

import numpy as np

from openalterego.research.somach_transfer import benchmark, load_folder
from openalterego.research.virtual_montage import write_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("overt", "mouthing", "exaggerated", "all"), default="all")
    args = parser.parse_args()
    phases = {"overt": "Phase_1_Overt", "mouthing": "Phase_3_Mouthing", "exaggerated": "Phase_5_Exaggerated"}
    selected = phases if args.phase == "all" else {args.phase: phases[args.phase]}
    results = {}
    for name, folder in selected.items():
        a = load_folder(args.data_root / "studyA_3phase" / folder)
        b = load_folder(args.data_root / "studyB_5phase" / folder)
        results[name] = benchmark(a[:2], b[:2])
        results[name]["skipped"] = {"configuration_a": a[2], "configuration_b": b[2]}
    measures = ("configuration_a_within_balanced_accuracy", "configuration_b_within_balanced_accuracy",
                "a_to_b_balanced_accuracy", "a_to_b_unlabeled_mean_shift_accuracy",
                "a_to_b_unlabeled_affine_match_accuracy")
    result = {"schema": "openalterego.cross-configuration-phases.v1", "phases": results,
              "mean_across_phases": {key: float(np.mean([value[key] for value in results.values()])) for key in measures}}
    result["source_commit"] = args.source_commit
    result["source_license"] = "CC BY 4.0, Somach-Systems-Inc/semg-silent-speech/LICENSE-DATA"
    write_report(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
