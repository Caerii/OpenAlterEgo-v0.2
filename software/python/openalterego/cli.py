"""OpenAlterEgo command line interface.

This is meant to be the *fast path* for:
- generating synthetic datasets
- training a baseline model
- serving realtime websocket output
- running a glasses/display stub

You can also run modules directly (python -m openalterego.api.server, etc.)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from . import __version__
from .sim.dataset import DatasetConfig, generate_dataset
from .sim.stream import ScenarioConfig, SimStreamConfig


def _cmd_sim_dataset(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(prog="openalterego sim-dataset")
    ap.add_argument("--out", type=str, required=True, help="Output directory (session folder)")
    ap.add_argument("--minutes", type=float, default=2.0)
    ap.add_argument("--fs", type=int, default=250)
    ap.add_argument("--channels", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--labels", type=str, default="yes,no,left,right,select,cancel")
    ap.add_argument("--p-event", type=float, default=0.65)
    args = ap.parse_args(argv)

    labels = [s.strip() for s in str(args.labels).split(",") if s.strip()]
    sc = ScenarioConfig(labels=labels, p_event=float(args.p_event))
    sim_cfg = SimStreamConfig(fs_hz=int(args.fs), channels=int(args.channels), seed=int(args.seed), scenario=sc, realtime_clock=False)
    ds = DatasetConfig(out_dir=Path(args.out), duration_s=float(args.minutes) * 60.0, config=sim_cfg)
    out = generate_dataset(ds)
    print(f"[openalterego] wrote dataset: {out}")
    print(f"  - signals.npy")
    print(f"  - events.csv")
    print(f"  - meta.json")
    return 0


def _delegate_module(module_main, argv: List[str]) -> int:
    """Run another module's main() with a temporary sys.argv."""
    old = sys.argv[:]
    try:
        sys.argv = [old[0]] + argv
        module_main()
        return 0
    finally:
        sys.argv = old


def main(argv: List[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    ap = argparse.ArgumentParser(prog="openalterego")
    ap.add_argument("--version", action="store_true", help="Print version and exit")

    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("sim-dataset", help="Generate a synthetic dataset session folder")

    sub.add_parser("serve", help="Run the websocket server (delegates to openalterego.api.server)")

    sub.add_parser("train", help="Train a baseline model (delegates to openalterego.ml.train)")

    sub.add_parser("glasses", help="Run the smart-glasses display stub (terminal)")

    ns, rest = ap.parse_known_args(argv)

    if ns.version:
        print(__version__)
        raise SystemExit(0)

    if ns.cmd == "sim-dataset":
        raise SystemExit(_cmd_sim_dataset(rest))

    if ns.cmd == "serve":
        from .api.server import main as server_main

        raise SystemExit(_delegate_module(server_main, rest))

    if ns.cmd == "train":
        from .ml.train import main as train_main

        raise SystemExit(_delegate_module(train_main, rest))

    if ns.cmd == "glasses":
        from .clients.glasses_stub import main as glasses_main

        raise SystemExit(_delegate_module(glasses_main, rest))

    ap.print_help()
    raise SystemExit(2)
