"""Cache publisher-verified raw event windows outside Git for neural controls."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from openalterego.research.signal_signature import event_windows
from openalterego.research.virtual_montage import sha256_file


EXPECTED_CHANNELS = [f"Ch_{index}_filt" for index in range(11)] + [
    f"Ch_{index}_filt" for index in (13, 14, 15)]


def cache_file(source: Path, expected_sha256: str, cache_dir: Path) -> Path:
    identity = hashlib.sha256((str(source.resolve()) + expected_sha256 +
                               "openalterego.raw-window-v1").encode()).hexdigest()[:20]
    return cache_dir / f"{source.stem}-{identity}.npz"


def extract(source: Path, expected_sha256: str, destination: Path) -> dict:
    import h5py

    if sha256_file(source) != expected_sha256:
        raise ValueError(f"input SHA-256 mismatch: {source}")
    with h5py.File(source, "r") as handle:
        group = handle["emg"]
        channels = [value.decode("ascii") for value in group["block1_items"][:]]
        if channels != EXPECTED_CHANNELS or list(group["block2_items"][:]) != [b"Label_int"]:
            raise ValueError(f"unexpected channel or label layout: {source}")
        signal = np.asarray(group["block1_values"][:], dtype=np.float32)
        labels = np.asarray(group["block2_values"][:, 0], dtype=np.int64)
    windows, event_labels = event_windows(signal, labels, width=400)
    if windows.shape[1:] != (400, 14) or not set(np.unique(event_labels)).issubset(set(range(9))):
        raise ValueError(f"invalid extracted windows: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as output:
        np.savez(output, windows=windows.astype(np.float32),
                 labels=event_labels.astype(np.int8))
    os.replace(temporary, destination)
    return {"source": source.name, "source_sha256": expected_sha256,
            "cache": destination.name, "events": len(event_labels),
            "shape": list(windows.shape)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", default=["S01", "S02", "S03"])
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    if not set(args.subjects).issubset({"S01", "S02", "S03"}):
        parser.error("S04 is excluded from development training")
    if args.cache_dir.resolve().is_relative_to(Path(__file__).resolve().parents[3]):
        parser.error("cache must stay outside the repository")
    for subject in args.subjects:
        manifest = json.loads((args.root / f"{subject}-silent-manifest.json").read_text())
        expected = {Path(row["path"]).name: row["sha256"]
                    for row in manifest["files"]}
        if len(expected) != 15:
            raise ValueError(f"{subject}: expected 15 recorded files")
        for name, sha256 in sorted(expected.items()):
            source = args.root / subject / "silent" / name
            destination = cache_file(source, sha256, args.cache_dir)
            if destination.exists():
                with np.load(destination, allow_pickle=False) as cached:
                    if cached["windows"].shape != (321, 400, 14) or len(cached["labels"]) != 321:
                        raise ValueError(f"invalid existing cache: {destination}")
                continue
            print(subject, extract(source, sha256, destination), flush=True)


if __name__ == "__main__":
    main()
