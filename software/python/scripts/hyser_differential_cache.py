"""Verify Hyser records and cache adjacent digital-differential features outside Git."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from openalterego.research.hyser_differential import (
    adjacent_pairs, differential_trial_features)
from openalterego.research.hyser_features import read_wfdb_16
from scripts.hyser_acquire import sha256_file


def build(root: Path, cache: Path, subject: str, session: int):
    root, cache = root.resolve(), cache.resolve()
    if cache.is_relative_to(Path(__file__).resolve().parents[3]):
        raise ValueError("raw-derived cache must stay outside repository")
    prefix = f"pr_dataset/subject{subject}_session{session}/"
    records = (root / "RECORDS").read_text().splitlines()
    bases = sorted((name for name in records if name.startswith(
        prefix + "dynamic_preprocess_sample")),
        key=lambda name: int(name.rsplit("sample", 1)[1]))
    labels_path = prefix + "label_dynamic.txt"
    labels = np.fromstring((root / labels_path).read_text(), sep=",", dtype=np.int8)
    if (len(labels) != len(bases) or len(labels) < 190 or
            [int(base.rsplit("sample", 1)[1]) for base in bases] !=
            list(range(1, len(bases) + 1))):
        raise ValueError("unexpected Hyser gesture records")
    expected = {name: digest for digest, name in
                (line.strip().split(" ", 1) for line in
                 (root / "SHA256SUMS.txt").read_text().splitlines())}
    names = [labels_path] + [name for base in bases
                             for name in (base + ".hea", base + ".dat")]
    identity = hashlib.sha256(json.dumps([(name, expected[name]) for name in names],
                                      separators=(",", ":")).encode()).hexdigest()
    destination = cache / f"subject{subject}-session{session}-{identity[:16]}.npz"
    metadata = destination.with_suffix(".json")
    if destination.exists() or metadata.exists():
        if not (destination.exists() and metadata.exists()):
            raise ValueError("partial differential cache exists")
        record = json.loads(metadata.read_text(encoding="utf-8"))
        if record.get("source_record_sha256") != identity:
            raise ValueError("differential cache provenance differs")
        with np.load(destination, allow_pickle=False) as cached:
            if (cached["features"].shape != (len(labels), 896) or
                    not np.array_equal(cached["labels"], labels)):
                raise ValueError("differential cache contents differ")
        print(subject, session, "cached", record["trials"], flush=True)
        return
    for name in names:
        if sha256_file(root / name) != expected[name]:
            raise ValueError(f"publisher hash mismatch: {name}")
    pair_map, site_names, features = None, None, []
    for base in bases:
        signal, sites = read_wfdb_16(root / base)
        if pair_map is None:
            site_names = sites
            pair_map = adjacent_pairs(sites)
        elif sites != site_names:
            raise ValueError(f"channel map changed: {base}")
        features.append(differential_trial_features(signal, pair_map))
    cache.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        np.savez_compressed(stream, features=np.stack(features), labels=labels)
    record = {"schema": "openalterego.hyser-digital-differential-features.v1",
              "subject": subject, "session": session,
              "scope": "Digitally subtracts publisher-preprocessed common-reference channels; does not reproduce analog differential noise or facial EMG",
              "source_record_sha256": identity,
              "publisher_manifest_sha256": sha256_file(root / "SHA256SUMS.txt"),
              "trials": len(labels), "contact_count": 256,
              "candidate_pair_count": len(pair_map),
              "site_names": site_names,
              "pairs": pair_map,
              "feature_formula": "log RMS and log waveform length of adjacent channel differences"}
    with metadata.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    print(subject, session, "extracted", len(labels), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--sessions", nargs="+", type=int, default=[1, 2])
    args = parser.parse_args()
    if (any(subject not in {f"{index:02d}" for index in range(1, 21)}
            for subject in args.subjects) or
            not set(args.sessions).issubset({1, 2})):
        parser.error("invalid subjects or sessions")
    for subject in args.subjects:
        for session in args.sessions:
            build(args.root, args.cache_dir, subject, session)


if __name__ == "__main__":
    main()
