"""Extract verified, time-resolved Hyser features outside Git."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from openalterego.research.hyser_features import read_wfdb_16, site_coordinates
from openalterego.research.hyser_temporal_features import temporal_trial_features
from scripts.hyser_acquire import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--baseline-cache", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--sessions", nargs="+", type=int, default=[1, 2])
    args = parser.parse_args()
    root = args.root.resolve()
    cache = args.cache_dir.resolve()
    if (cache.is_relative_to(Path(__file__).resolve().parents[3]) or
            any(subject not in {f"{index:02d}" for index in range(1, 21)}
                for subject in args.subjects) or
            not set(args.sessions).issubset({1, 2})):
        parser.error("invalid output, subject or session")
    sums = root / "SHA256SUMS.txt"
    expected = {name: digest for digest, name in
                (line.strip().split(" ", 1) for line in sums.read_text().splitlines())}
    records = (root / "RECORDS").read_text().splitlines()
    cache.mkdir(parents=True, exist_ok=True)
    for subject in args.subjects:
        for session in args.sessions:
            prefix = f"pr_dataset/subject{subject}_session{session}/"
            bases = [base for base in records
                     if base.startswith(prefix + "dynamic_preprocess_sample")]
            bases.sort(key=lambda base: int(base.rsplit("sample", 1)[1]))
            labels_path = prefix + "label_dynamic.txt"
            labels = np.fromstring((root / labels_path).read_text().strip(),
                                   sep=",", dtype=np.int64)
            if (len(labels) != len(bases) or
                    [int(base.rsplit("sample", 1)[1]) for base in bases] !=
                    list(range(1, len(labels) + 1))):
                raise ValueError(f"invalid trial sequence: {prefix}")
            source_names = [labels_path] + [name for base in bases
                                            for name in (base + ".hea", base + ".dat")]
            source_record = [(name, expected[name]) for name in source_names]
            identity = hashlib.sha256(json.dumps(source_record, separators=(",", ":")).encode()).hexdigest()
            destination = cache / f"subject{subject}-session{session}-{identity[:16]}.npz"
            metadata_path = destination.with_suffix(".json")
            if destination.exists() and metadata_path.exists():
                with np.load(destination, allow_pickle=False) as data:
                    if data["features"].shape != (len(bases), 2560):
                        raise ValueError("unexpected temporal cache shape")
                if json.loads(metadata_path.read_text())["source_record_sha256"] != identity:
                    raise ValueError("temporal cache source hash mismatch")
                print(subject, session, "cached", len(bases), flush=True)
                continue
            baseline_paths = list(args.baseline_cache.glob(
                f"subject{subject}-session{session}-{identity[:16]}.npz"))
            if len(baseline_paths) != 1:
                raise ValueError("matching verified baseline cache missing")
            with np.load(baseline_paths[0], allow_pickle=False) as baseline:
                baseline_features = baseline["features"]
                baseline_labels = baseline["labels"]
            if baseline_features.shape != (len(bases), 512) or not np.array_equal(
                    baseline_labels, labels):
                raise ValueError("baseline feature cache differs")
            for name, digest in source_record:
                if sha256_file(root / name) != digest:
                    raise ValueError(f"publisher hash mismatch: {name}")
            features = []
            site_names = None
            for index, base in enumerate(bases):
                signal, sites = read_wfdb_16(root / base)
                if site_names is None:
                    site_names = sites
                    site_coordinates(sites)
                elif sites != site_names:
                    raise ValueError("channel order changed")
                feature = temporal_trial_features(signal)
                np.testing.assert_array_equal(
                    feature.reshape(256, 10)[:, :2].reshape(512), baseline_features[index])
                features.append(feature)
            temporary = destination.with_name(f"{destination.name}.{os.getpid()}.tmp")
            with temporary.open("wb") as output:
                np.savez_compressed(output, features=np.stack(features),
                                    labels=labels.astype(np.int8),
                                    sample_ids=np.arange(1, len(labels) + 1, dtype=np.int16))
            os.replace(temporary, destination)
            record = {"schema": "openalterego.hyser-temporal-trial-features.v1",
                      "subject": subject, "session": session,
                      "source_record_sha256": identity,
                      "publisher_manifest_sha256": sha256_file(sums),
                      "site_names": site_names, "trials": len(features),
                      "features_per_site": 10,
                      "feature_formula": "whole-trial plus four ordered 512-sample bins; each log RMS and log waveform length",
                      "baseline_whole_trial_exact_parity": True}
            metadata_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            print(subject, session, "extracted", len(features), flush=True)


if __name__ == "__main__":
    main()
