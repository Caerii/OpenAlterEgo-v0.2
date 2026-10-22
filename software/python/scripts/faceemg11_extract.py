"""Extract trial-local facial EMG features from the public FaceEMG-11 archive."""

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np


EXPECTED_ARCHIVE_MD5 = "a7308c74a23731e57bec2e2b7338315d"


def trial_features(data: np.ndarray) -> np.ndarray:
    """Interleaved log RMS and waveform length per recorded channel."""
    if (data.shape != (330, 20, 1500) or data.dtype != np.float32 or
            not np.isfinite(data).all()):
        raise ValueError("unexpected FaceEMG-11 trial tensor")
    signal = data.astype(np.float64)
    rms = np.sqrt(np.mean(np.square(signal), axis=2))
    waveform = np.mean(np.abs(np.diff(signal, axis=2)), axis=2)
    return np.stack((np.log(np.maximum(rms, 1e-8)),
                     np.log(np.maximum(waveform, 1e-8))),
                    axis=2).reshape(len(data), -1).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.archive.is_file():
        parser.error("archive does not exist")
    digest = hashlib.md5()
    with args.archive.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != EXPECTED_ARCHIVE_MD5:
        raise ValueError("archive checksum differs from Zenodo")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive) as archive:
        for subject in range(1, 13):
            stem = f"sub-{subject:02d}_trials"
            target = args.output_dir / (stem + "-features.npz")
            metadata_path = args.output_dir / (stem + "-features.json")
            if target.exists() or metadata_path.exists():
                raise ValueError(f"feature cache already exists: {subject:02d}")
            member = f"FaceEMG-11/{stem}.npz"
            payload = archive.read(member)
            with np.load(io.BytesIO(payload), allow_pickle=False) as source:
                data = source["data"]
                labels = source["labels"]
                groups = source["groups"]
                fs = source["fs"]
                channel_names = source["ch_names"]
                if (labels.shape != (330,) or groups.shape != (330,) or
                        float(fs) != 1000 or len(channel_names) != 20 or
                        not np.array_equal(np.unique(labels), np.arange(1, 12)) or
                        not np.array_equal(np.unique(groups), np.arange(1, 31))):
                    raise ValueError(f"unexpected trial metadata: {subject:02d}")
                for block in range(1, 31):
                    if not np.array_equal(np.sort(labels[groups == block]),
                                          np.arange(1, 12)):
                        raise ValueError(f"block incomplete: {subject:02d}/{block}")
                features = trial_features(data)
            np.savez_compressed(target, features=features,
                                labels=labels, groups=groups)
            metadata_path.write_text(json.dumps({
                "schema": "openalterego.faceemg11-features.v1",
                "source": "https://doi.org/10.5281/zenodo.22137268",
                "archive_md5": digest.hexdigest(),
                "member": member,
                "member_sha256": hashlib.sha256(payload).hexdigest(),
                "features": "trial-local log RMS and log waveform length, 2 per channel",
                "shape": list(features.shape),
                "channel_names": channel_names.tolist(),
            }, indent=2) + "\n", encoding="utf-8")
            print(stem, features.shape, flush=True)


if __name__ == "__main__":
    main()
