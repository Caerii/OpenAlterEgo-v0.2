"""Extract four local time-bin features per FaceEMG-11 trial and channel."""

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np


ARCHIVE_MD5 = "a7308c74a23731e57bec2e2b7338315d"
BINS = 4


def four_bin_features(data: np.ndarray) -> np.ndarray:
    """Four bins of log RMS and log waveform length, interleaved by channel."""
    if (data.shape != (330, 20, 1500) or data.dtype != np.float32 or
            not np.isfinite(data).all()):
        raise ValueError("unexpected facial trial tensor")
    signal = data.reshape(330, 20, BINS, 375).astype(np.float64)
    rms = np.sqrt(np.mean(np.square(signal), axis=3))
    waveform = np.mean(np.abs(np.diff(signal, axis=3)), axis=3)
    return np.stack((np.log(np.maximum(rms, 1e-8)),
                     np.log(np.maximum(waveform, 1e-8))),
                    axis=3).reshape(330, 20 * BINS * 2).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    digest = hashlib.md5()
    with args.archive.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != ARCHIVE_MD5:
        raise ValueError("Zenodo archive checksum differs")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive) as archive:
        for subject in range(1, 13):
            stem = f"sub-{subject:02d}_trials"
            destination = args.output_dir / (stem + "-temporal.npz")
            metadata_path = args.output_dir / (stem + "-temporal.json")
            if destination.exists() or metadata_path.exists():
                raise ValueError(f"cache exists: {subject:02d}")
            member = f"FaceEMG-11/{stem}.npz"
            payload = archive.read(member)
            with np.load(io.BytesIO(payload), allow_pickle=False) as source:
                labels = source["labels"]
                groups = source["groups"]
                if (labels.shape != (330,) or groups.shape != (330,) or
                        not np.array_equal(np.unique(groups), np.arange(1, 31))):
                    raise ValueError(f"unexpected trial order: {subject:02d}")
                features = four_bin_features(source["data"])
            np.savez_compressed(destination, features=features,
                                labels=labels, groups=groups)
            metadata_path.write_text(json.dumps({
                "schema": "openalterego.faceemg11-four-bin.v1",
                "source": "https://doi.org/10.5281/zenodo.22137268",
                "source_member_sha256": hashlib.sha256(payload).hexdigest(),
                "archive_md5": digest.hexdigest(),
                "features": "four consecutive 375-sample bins, log RMS and log waveform length each",
                "shape": list(features.shape),
            }, indent=2) + "\n", encoding="utf-8")
            print(stem, features.shape, flush=True)


if __name__ == "__main__":
    main()
