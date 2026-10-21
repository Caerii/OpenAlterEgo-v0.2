"""Stream a bounded, allowlisted paired-EMG/audio subset from the Gaddy archive.

The input archive must be obtained from Zenodo 4064409 and verified against
its published MD5 before use. Raw data and this output stay outside Git.
"""

import argparse
import hashlib
import json
import re
import tarfile
from pathlib import Path


PUBLISHED_MD5 = "7f97d2182b896652999b1b2d0c69fd7b"
PREFIX = "emg_data/closed_vocab/voiced/5-19/"
SUFFIXES = ("emg.npy", "audio_clean.flac", "info.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()
    if args.count < 1 or args.count > 100:
        parser.error("count must be between 1 and 100")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("output must be a new or empty directory")
    digest = hashlib.md5()
    with args.archive.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != PUBLISHED_MD5:
        parser.error("archive differs from publisher MD5")
    args.out.mkdir(parents=True, exist_ok=True)
    wanted = {f"{sample}_{suffix}" for sample in range(args.count) for suffix in SUFFIXES}
    manifest = {}
    with tarfile.open(args.archive, "r|gz") as archive:
        for member in archive:
            if not member.name.startswith(PREFIX) or not member.isfile():
                continue
            name = member.name[len(PREFIX):]
            if name not in wanted or not re.fullmatch(r"[0-9]+_(emg\.npy|audio_clean\.flac|info\.json)", name):
                continue
            if member.size < 1 or member.size > 5_000_000:
                raise ValueError(f"unexpected member size: {name}")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"unreadable member: {name}")
            payload = source.read(member.size + 1)
            if len(payload) != member.size:
                raise ValueError(f"truncated member: {name}")
            if name.endswith("info.json") and not isinstance(json.loads(payload)["text"], str):
                raise ValueError(f"invalid metadata: {name}")
            with (args.out / name).open("xb") as target:
                target.write(payload)
            manifest[name] = {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
            wanted.remove(name)
            if not wanted:
                break
    if wanted:
        raise ValueError(f"archive lacks {len(wanted)} requested components")
    with (args.out / "manifest.json").open("x", encoding="utf-8") as output:
        json.dump({"schema": "openalterego.gaddy-voiced-subset.v1", "source_md5": PUBLISHED_MD5,
                   "source_path": PREFIX, "files": manifest}, output, indent=2, sort_keys=True)
        output.write("\n")
    print(f"extracted {args.count} matched voiced examples to {args.out}")


if __name__ == "__main__":
    main()
