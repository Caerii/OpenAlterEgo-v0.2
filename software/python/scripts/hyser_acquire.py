"""Download a publisher-hash-verified Hyser dynamic PR subset outside Git."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


BASE = "https://physionet-open.s3.amazonaws.com/hd-semg/2.0.0/"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def acquire(root: Path, name: str, expected: str) -> tuple[str, int, bool]:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("file escapes dataset root")
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("file escapes dataset root")
    if path.exists():
        if sha256_file(path) != expected:
            raise ValueError(f"existing file hash mismatch: {path}")
        return name, path.stat().st_size, True
    temporary = path.with_name(f"{path.name}.{os.getpid()}.part")
    for attempt in range(5):
        try:
            digest = hashlib.sha256()
            with urlopen(BASE + name, timeout=90) as response, temporary.open("wb") as output:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    digest.update(chunk)
                    output.write(chunk)
            if digest.hexdigest() != expected:
                raise ValueError(f"download SHA-256 mismatch: {name}")
            os.replace(temporary, path)
            return name, path.stat().st_size, False
        except (HTTPError, URLError, TimeoutError, ConnectionError) as error:
            if (isinstance(error, HTTPError) and error.code not in (429, 500, 502, 503, 504)) or attempt == 4:
                raise
            time.sleep(min(2 ** attempt, 16))
        finally:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--sessions", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if (any(subject not in {f"{index:02d}" for index in range(1, 21)}
            for subject in args.subjects) or
            not set(args.sessions).issubset({1, 2}) or
            args.max_samples < 0 or not 1 <= args.workers <= 32):
        parser.error("invalid subject, session, sample limit or worker count")
    root = args.root.resolve()
    if root.is_relative_to(Path(__file__).resolve().parents[3]):
        parser.error("raw dataset must remain outside Git")
    sums = root / "SHA256SUMS.txt"
    records = root / "RECORDS"
    if not sums.exists() or not records.exists():
        parser.error("download official SHA256SUMS.txt and RECORDS into --root first")
    # Several publisher files have identical contents and hashes. Index by
    # pathname directly so a duplicate digest cannot discard a record.
    expected = {name: digest for digest, name in
                (line.strip().split(" ", 1) for line in sums.read_text().splitlines())}
    all_records = records.read_text().splitlines()
    names = []
    counts = {}
    for subject in args.subjects:
        for session in args.sessions:
            prefix = f"pr_dataset/subject{subject}_session{session}/"
            bases = [base for base in all_records
                     if base.startswith(prefix + "dynamic_preprocess_sample")]
            bases.sort(key=lambda base: int(base.rsplit("sample", 1)[1]))
            if args.max_samples:
                bases = bases[:args.max_samples]
            if not bases:
                raise ValueError(f"no publisher records for {prefix}")
            counts[f"{subject}/{session}"] = len(bases)
            names.append(prefix + "label_dynamic.txt")
            for base in bases:
                names.extend((base + ".hea", base + ".dat"))
    if any(name not in expected for name in names):
        raise ValueError("record missing from publisher SHA256SUMS")
    bytes_total = 0
    cached = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(acquire, root, name, expected[name]): name for name in names}
        for index, future in enumerate(as_completed(jobs), 1):
            name, size, hit = future.result()
            bytes_total += size
            cached += int(hit)
            if index % 100 == 0 or index == len(names):
                print(f"verified {index}/{len(names)} files; {bytes_total / 1e9:.2f} GB", flush=True)
    print(json.dumps({"schema": "openalterego.hyser-acquisition.v1",
                      "publisher_manifest_sha256": sha256_file(sums),
                      "records_sha256": sha256_file(records),
                      "record_counts": counts, "files": len(names),
                      "cached_files": cached, "verified_bytes": bytes_total}, indent=2))


if __name__ == "__main__":
    main()
