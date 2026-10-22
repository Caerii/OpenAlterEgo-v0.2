"""Prepare or execute an auditable Gowda sim-to-real transfer experiment.

This wrapper keeps generated outputs outside input sessions. It records exact
input bytes and process outcome; it does not confer scientific acceptance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


INPUT_FILES = ("signals.npy", "events.csv", "meta.json")
ROOT = Path(__file__).resolve().parents[3]
PYTHON_DIR = ROOT / "software" / "python"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def session_record(path: Path, *, real: bool) -> dict:
    path = path.resolve(strict=True)
    files = {}
    for name in INPUT_FILES:
        item = path / name
        if not item.is_file():
            raise ValueError(f"missing required input: {item}")
        files[name] = {"sha256": sha256_file(item), "bytes": item.stat().st_size}
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    if int(meta.get("fs_hz", 0)) != 5000:
        raise ValueError(f"Gowda transfer requires 5000 Hz: {path}")
    if int(meta.get("channels", 0)) != 31:
        raise ValueError(f"Gowda transfer requires 31 channels: {path}")
    trial_ids = set()
    with (path / "events.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "trial_id" not in reader.fieldnames:
            raise ValueError(f"events.csv lacks trial_id: {path}")
        for row in reader:
            trial_ids.add(int(row["trial_id"]))
    if not trial_ids or min(trial_ids) < 0:
        raise ValueError(f"empty or invalid trial IDs: {path}")
    split = {
        "train_trials": sum(t < 370 for t in trial_ids),
        "validation_trials": sum(370 <= t < 400 for t in trial_ids),
        "test_trials": sum(400 <= t < 500 for t in trial_ids),
    }
    if real and min(split.values()) == 0:
        raise ValueError(f"real session must contain train, validation, and test trials: {path}")
    return {
        "path": str(path), "files": files, "fs_hz": 5000, "channels": 31,
        "split": split,
        "official_split_complete": split["train_trials"] == 370
        and split["validation_trials"] == 30
        and split["test_trials"] in (99, 100),
        "simulation": {
            key: meta.get(key) for key in (
                "sim_engine", "realism_preset", "sim_biophysical_model", "hardware_spec"
            ) if key in meta
        },
    }


def git_state() -> dict:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    return {"commit": commit, "dirty": bool(status.strip())}


def write_manifest(path: Path, payload: dict) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def prepare(sim: Path, real: Path, out: Path, *, device: str, seed: int) -> tuple[dict, list[str]]:
    sim = sim.resolve(strict=True)
    real = real.resolve(strict=True)
    out = out.resolve()
    if out == sim or out == real or sim in out.parents or real in out.parents:
        raise ValueError("output directory must be outside both input sessions")
    if (out / "manifest.json").exists():
        raise ValueError(f"experiment already exists: {out}")
    sim_record = session_record(sim, real=False)
    real_record = session_record(real, real=True)
    command = [
        "uv", "run", "--frozen", "openalterego", "analyze", "sim-transfer",
        "--sim", str(sim), "--real", str(real), "--device", device,
        "--seed", str(seed), "--real-fracs", "0,0.1,0.5,1.0",
        "--work-dir", str(out / "work"), "--out", str(out / "result.json"),
    ]
    manifest = {
        "schema_version": 1,
        "status": "prepared",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Gowda small-vocabulary sim-to-real transfer; not wearable validation",
        "code": git_state(),
        "lock_sha256": sha256_file(PYTHON_DIR / "uv.lock"),
        "inputs": {"sim": sim_record, "real": real_record},
        "parameters": {"device": device, "seed": seed, "real_fracs": [0, 0.1, 0.5, 1.0]},
        "environment": {"python": sys.version.split()[0], "platform": platform.platform()},
        "command": command,
    }
    out.mkdir(parents=True, exist_ok=True)
    write_manifest(out / "manifest.json", manifest)
    return manifest, command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim", type=Path, required=True)
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--execute", action="store_true", help="Run training after preparing the manifest")
    args = parser.parse_args()
    out = args.out.resolve()
    created = False
    try:
        manifest, command = prepare(args.sim, args.real, out, device=args.device, seed=args.seed)
        created = True
        if not args.execute:
            print(out / "manifest.json")
            return 0
        if manifest["code"]["dirty"]:
            raise ValueError("refusing to execute from a dirty checkout; commit the experiment code first")
        if not manifest["inputs"]["real"]["official_split_complete"]:
            raise ValueError("refusing benchmark execution without the 370/30/99-or-100 real split")
        manifest["status"] = "running"
        write_manifest(out / "manifest.json", manifest)
        with (out / "stdout.log").open("w", encoding="utf-8") as stdout, (
            out / "stderr.log"
        ).open("w", encoding="utf-8") as stderr:
            process = subprocess.run(command, cwd=PYTHON_DIR, stdout=stdout, stderr=stderr, check=False)
        manifest["exit_code"] = process.returncode
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["status"] = "completed" if process.returncode == 0 and (
            out / "result.json"
        ).is_file() else "failed"
        if (out / "result.json").is_file():
            manifest["result_sha256"] = sha256_file(out / "result.json")
        manifest["checkpoints"] = {
            str(path.relative_to(out)): sha256_file(path)
            for path in sorted((out / "work").glob("*.pt"))
        }
        write_manifest(out / "manifest.json", manifest)
        print(out / "manifest.json")
        return process.returncode if manifest["status"] == "completed" else 2
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        if created:
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            manifest["status"] = "failed_preflight" if manifest["status"] == "prepared" else "failed"
            manifest["error"] = str(exc)
            write_manifest(out / "manifest.json", manifest)
        print(f"transfer experiment: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
