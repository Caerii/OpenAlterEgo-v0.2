"""Build verified trial features for Hyser dynamic gestures outside Git."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from openalterego.research.hyser_features import read_wfdb_16, site_coordinates, trial_features
from scripts.hyser_acquire import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--sessions", nargs="+", type=int, default=[1, 2])
    args = parser.parse_args()
    root = args.root.resolve()
    cache = args.cache_dir.resolve()
    if (cache.is_relative_to(Path(__file__).resolve().parents[3]) or
            any(subject not in {f"{index:02d}" for index in range(1, 21)}
                for subject in args.subjects) or
            not set(args.sessions).issubset({1, 2})):
        parser.error("invalid cache path, subjects or sessions")
    sums = root / "SHA256SUMS.txt"
    records = root / "RECORDS"
    expected = {name: digest for digest, name in
                (line.strip().split(" ", 1) for line in sums.read_text().splitlines())}
    all_records = records.read_text().splitlines()
    cache.mkdir(parents=True, exist_ok=True)
    for subject in args.subjects:
        for session in args.sessions:
            prefix = f"pr_dataset/subject{subject}_session{session}/"
            bases = [base for base in all_records
                     if base.startswith(prefix + "dynamic_preprocess_sample")]
            bases.sort(key=lambda base: int(base.rsplit("sample", 1)[1]))
            labels_path = prefix + "label_dynamic.txt"
            labels = np.fromstring((root / labels_path).read_text().strip(),
                                   sep=",", dtype=np.int64)
            if (len(labels) != len(bases) or
                    len(np.unique(labels)) < 30 or
                    not set(np.unique(labels)).issubset(set(range(1, 35))) or
                    [int(base.rsplit("sample", 1)[1]) for base in bases] !=
                    list(range(1, len(labels) + 1))):
                raise ValueError(f"unexpected gesture labels: {prefix}")
            source_names = [labels_path] + [name for base in bases
                                            for name in (base + ".hea", base + ".dat")]
            source_record = [(name, expected[name]) for name in source_names]
            identity = hashlib.sha256(json.dumps(source_record, separators=(",", ":")).encode()).hexdigest()
            destination = cache / f"subject{subject}-session{session}-{identity[:16]}.npz"
            metadata = destination.with_suffix(".json")
            if destination.exists() and metadata.exists():
                with np.load(destination, allow_pickle=False) as data:
                    if data["features"].shape != (len(bases), 512):
                        raise ValueError(f"invalid cached features: {destination}")
                if json.loads(metadata.read_text())["source_record_sha256"] != identity:
                    raise ValueError(f"cache metadata mismatch: {metadata}")
                print(subject, session, "cached", len(bases), flush=True)
                continue
            for name, digest in source_record:
                if sha256_file(root / name) != digest:
                    raise ValueError(f"publisher hash mismatch: {name}")
            features, selected_labels, sample_ids = [], [], []
            site_names = None
            for base in bases:
                signal, sites = read_wfdb_16(root / base)
                if site_names is None:
                    site_names = sites
                    site_coordinates(sites)
                elif sites != site_names:
                    raise ValueError(f"channel order differs: {base}")
                index = int(base.rsplit("sample", 1)[1])
                if index < 1 or index > len(labels):
                    raise ValueError(f"sample index outside label list: {base}")
                features.append(trial_features(signal))
                selected_labels.append(int(labels[index - 1]))
                sample_ids.append(index)
            if not features:
                raise ValueError(f"no trials: {prefix}")
            temporary = destination.with_name(f"{destination.name}.{os.getpid()}.tmp")
            with temporary.open("wb") as output:
                np.savez_compressed(output, features=np.stack(features),
                                    labels=np.asarray(selected_labels, dtype=np.int8),
                                    sample_ids=np.asarray(sample_ids, dtype=np.int16))
            os.replace(temporary, destination)
            record = {"schema": "openalterego.hyser-trial-features.v1",
                      "subject": subject, "session": session,
                      "source_record_sha256": identity,
                      "publisher_manifest_sha256": sha256_file(sums),
                      "site_names": site_names, "trials": len(features),
                      "nominal_204_trial_shortfall": 204 - len(sample_ids),
                      "feature_formula": "whole-trial log RMS and log waveform length of WFDB-scaled 256 channels"}
            metadata.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            print(subject, session, "extracted", len(features), flush=True)


if __name__ == "__main__":
    main()
