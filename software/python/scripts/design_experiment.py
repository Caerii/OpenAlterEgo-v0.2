"""Create or score prospective wearable design experiments."""

import argparse
import json
from pathlib import Path

from openalterego.research.design_experiment import register, score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    registration = subcommands.add_parser("register")
    registration.add_argument("protocol", type=Path)
    registration.add_argument("preregistration", type=Path)
    scoring = subcommands.add_parser("score")
    scoring.add_argument("preregistration", type=Path)
    scoring.add_argument("outcomes", type=Path)
    scoring.add_argument("report", type=Path)
    args = parser.parse_args()
    if args.command == "register":
        value = register(json.loads(args.protocol.read_text(encoding="utf-8")), args.preregistration)
        print(f"registered {value['protocol']['experiment_id']} {value['protocol_sha256']}")
    else:
        prereg = json.loads(args.preregistration.read_text(encoding="utf-8"))
        outcomes = json.loads(args.outcomes.read_text(encoding="utf-8"))
        result = score(prereg, outcomes)
        with args.report.open("x", encoding="utf-8") as output:
            json.dump(result, output, indent=2, sort_keys=True)
            output.write("\n")
        print(f"scored {result['experiment_id']} regret={result['decision_regret']:.4f}")


if __name__ == "__main__":
    main()
