"""Download a size-bounded, SHA-256-verified public SilentWear subset."""

import argparse
import hashlib
import json
import os
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


DATASET = "PulpBio/SilentWear"
REVISION = "9fd2d8f5f8e51333b5f3314d878163710be2b3b2"
LICENSE = "Apache-2.0"  # Dataset card at the pinned revision; preserve its attribution.


def fetch_json(url: str) -> object:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def download(file: dict, root: Path) -> dict:
    relative = file["path"]
    expected = file["lfs"]["oid"]
    size = file["size"]
    target = root / Path(*Path(relative).parts[1:])
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size != size or digest(target) != expected:
            raise ValueError(f"existing file does not match publisher hash: {target}")
        return {"path": relative, "size": size, "sha256": expected, "source": "existing_verified"}
    temp = target.with_suffix(target.suffix + ".download")
    if temp.exists():
        raise FileExistsError(f"incomplete download exists; inspect it first: {temp}")
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{urllib.parse.quote(relative)}?download=true"
    try:
        with urllib.request.urlopen(url, timeout=120) as response, temp.open("xb") as output:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        if temp.stat().st_size != size or digest(temp) != expected:
            raise ValueError(f"publisher size or SHA-256 mismatch: {relative}")
        temp.rename(target)
    except Exception:
        temp.unlink(missing_ok=True)
        raise
    return {"path": relative, "size": size, "sha256": expected, "source": "downloaded"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="External dataset directory, outside the Git checkout")
    parser.add_argument("--subject", default="S01")
    parser.add_argument("--condition", default="silent", choices=("silent", "vocalized"))
    parser.add_argument("--max-gb", type=float, default=1.0)
    args = parser.parse_args()
    if not args.subject.startswith("S") or len(args.subject) != 3 or not args.subject[1:].isdigit():
        parser.error("subject must look like S01")
    if args.max_gb <= 0:
        parser.error("max-gb must be positive")
    prefix = f"data_raw_and_filt/{args.subject}/{args.condition}"
    listing_url = f"https://huggingface.co/api/datasets/{DATASET}/tree/{REVISION}/{prefix}?recursive=false&expand=true"
    listing = fetch_json(listing_url)
    selected = sorted((item for item in listing if item.get("type") == "file" and item["path"].endswith(".h5")), key=lambda item: item["path"])
    if not selected or any("lfs" not in item for item in selected):
        raise ValueError("publisher listing has no hashed HDF5 files")
    total = sum(item["size"] for item in selected)
    if total > args.max_gb * 1_000_000_000:
        raise ValueError(f"planned download {total} bytes exceeds --max-gb {args.max_gb}")
    print(f"{len(selected)} files, {total} bytes, pinned revision {REVISION}", flush=True)
    files = []
    for item in selected:
        result = download(item, args.out)
        files.append(result)
        print(result["source"], result["path"], flush=True)
    manifest = {
        "schema": "openalterego.public-data-import.v1",
        "dataset": DATASET,
        "revision": REVISION,
        "license": LICENSE,
        "listing_url": listing_url,
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "total_bytes": total,
        "files": files,
    }
    manifest_path = args.out / f"{args.subject}-{args.condition}-manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"manifest already exists: {manifest_path}")
    with manifest_path.open("x", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2, sort_keys=True)
        output.write("\n")
    print(manifest_path)


if __name__ == "__main__":
    main()
