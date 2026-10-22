"""Search feasible virtual SilentWear montages without opening S04."""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from openalterego.research.program_search import search
from openalterego.research.virtual_montage import load_silentwear_events, sha256_file


def load_batch(path: Path, expected_sha256: str, cache_dir: Path):
    repo = Path(__file__).resolve().parents[3]
    if cache_dir.resolve().is_relative_to(repo):
        raise ValueError("raw-derived feature cache must stay outside the repository")
    stat = path.stat()
    digest = hashlib.sha256()
    digest.update(b"openalterego.silentwear-event-cache.v1\0")
    digest.update(str(path.resolve()).encode())
    digest.update(f"{stat.st_size}:{stat.st_mtime_ns}:{expected_sha256}".encode())
    digest.update(Path(load_silentwear_events.__code__.co_filename).read_bytes())
    cache_file = cache_dir.resolve() / f"{path.stem}-{digest.hexdigest()[:24]}.npz"
    if cache_file.exists():
        with np.load(cache_file, allow_pickle=False) as cached:
            features, labels = cached["features"], cached["labels"]
        hit = True
    else:
        if sha256_file(path) != expected_sha256:
            raise ValueError(f"input SHA-256 mismatch: {path}")
        features, labels, _ = load_silentwear_events(path)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_file.with_name(f"{cache_file.name}.{os.getpid()}.tmp")
        with temporary.open("wb") as output:
            np.savez(output, features=features, labels=labels)
        os.replace(temporary, cache_file)
        hit = False
    if (features.ndim != 2 or features.shape[1] != 28 or
            labels.ndim != 1 or len(features) != len(labels) or
            not np.isfinite(features).all() or
            not set(np.unique(labels)).issubset(set(range(9)))):
        raise ValueError(f"invalid cached events: {cache_file}")
    return (features, labels), hit


def subject_sessions(root: Path, subject: str, cache_dir: Path):
    manifest_path = root / f"{subject}-silent-manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    expected = {Path(entry["path"]).name: entry["sha256"]
                for entry in manifest["files"]}
    if len(expected) != 15:
        raise ValueError(f"{subject}: expected 15 manifest entries")
    provenance = []

    def load_session(session: int):
        paths = sorted((root / subject / "silent").glob(f"sess_{session}_batch_*.h5"))
        if len(paths) != 5:
            raise ValueError(f"{subject}: expected five batches in session {session}")
        rows = []
        for path in paths:
            row, hit = load_batch(path, expected[path.name], cache_dir)
            rows.append(row)
            provenance.append({"session": session, "file": path.name,
                               "source_sha256": expected[path.name],
                               "verified_on_cache_miss": not hit,
                               "event_cache_hit": hit})
        return rows

    return load_session, provenance, hashlib.sha256(manifest_bytes).hexdigest()


def register_selection(path: Path, selection: dict, subject: str,
                       input_manifest_sha256: str) -> str:
    canonical = json.dumps(selection, sort_keys=True,
                           separators=(",", ":"), allow_nan=False).encode()
    selection_sha256 = hashlib.sha256(canonical).hexdigest()
    code_sha256 = hashlib.sha256(Path(search.__code__.co_filename).read_bytes()).hexdigest()
    identity = {"subject": subject, "input_manifest_sha256": input_manifest_sha256,
                "code_sha256": code_sha256, "selection_sha256": selection_sha256}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if (existing.get("schema") != "openalterego.virtual-program-selection-record.v1"
                or any(existing.get(key) != value for key, value in identity.items())
                or existing.get("selection") != selection):
            raise ValueError(f"existing selection record differs: {path}")
        return selection_sha256
    record = {"schema": "openalterego.virtual-program-selection-record.v1",
              "registered_utc": datetime.now(timezone.utc).isoformat(),
              **identity, "selection": selection,
              "scope": "Local exclusive record, not an external immutable timestamp."}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        json.dump(record, output, indent=2, allow_nan=False)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    return selection_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", default=["S01", "S02", "S03"])
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--stage-one-per-strategy", type=int, default=16)
    parser.add_argument("--finalists-per-strategy", type=int, default=4)
    parser.add_argument("--proxy", choices=("ridge", "centroid"), default="ridge")
    parser.add_argument("--selection-mode", choices=("one_way", "bidirectional"),
                        default="one_way")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true",
                        help="continue a matching partial cohort result")
    args = parser.parse_args()
    if not set(args.subjects).issubset({"S01", "S02", "S03"}):
        parser.error("S04 is sealed; search may use S01, S02 and S03 only")
    config = {"root": str(args.root.resolve()), "subjects": args.subjects,
              "seed": args.seed,
              "proxy": args.proxy,
              "selection_mode": args.selection_mode,
              "stage_one_per_strategy": args.stage_one_per_strategy,
              "finalists_per_strategy": args.finalists_per_strategy}
    if args.output.exists():
        if not args.resume:
            parser.error("output exists; use --resume only for a matching run")
        result = json.loads(args.output.read_text(encoding="utf-8"))
        if (result.get("schema") != "openalterego.virtual-program-search-cohort.v5"
                or result.get("config") != config):
            parser.error("existing result does not match this search configuration")
    else:
        result = {"schema": "openalterego.virtual-program-search-cohort.v5",
                  "config": config, "subjects": {},
                  "scope": "Virtual subsets of existing recorded channels, not new electrode positions."}
    for subject in args.subjects:
        if subject in result["subjects"]:
            print(f"{subject} already recorded", flush=True)
            continue
        load_session, provenance, manifest_hash = subject_sessions(
            args.root, subject, args.cache_dir)
        train, validation = load_session(1), load_session(2)
        selection_path = args.output.with_name(
            f"{args.output.stem}-{subject}-selection.json")
        receipt = {}

        def freeze(selection):
            receipt["selection_sha256"] = register_selection(
                selection_path, selection, subject, manifest_hash)

        subject_result = search(train, validation, lambda: load_session(3),
                                stage_one_per_strategy=args.stage_one_per_strategy,
                                finalists_per_strategy=args.finalists_per_strategy,
                                proxy_kind=args.proxy,
                                selection_mode=args.selection_mode,
                                seed=args.seed, on_selection=freeze)
        subject_result["inputs"] = provenance
        subject_result["selection_record"] = str(selection_path)
        subject_result["selection_sha256"] = receipt["selection_sha256"]
        result["subjects"][subject] = subject_result
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f"{args.output.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                             encoding="utf-8")
        os.replace(temporary, args.output)
        choices = {name: row["test_balanced_accuracy"] for name, row in
                   subject_result["strategies"].items()}
        print(subject, choices, flush=True)


if __name__ == "__main__":
    main()
