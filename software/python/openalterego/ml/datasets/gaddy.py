"""Gaddy silent-speech EMG (Zenodo 10.5281/zenodo.4064409) import adapter.

Raw layout after extracting ``emg_data.tar.gz``:
  {id}_emg.npy      float array (T, 8) @ 1000 Hz
  {id}_info.json    metadata incl. text, sentence_index, silent

Reference samples have ``sentence_index: -1`` and are skipped.
"""

from __future__ import annotations

import json
import re
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

from .session import SessionMeta, write_session_folder

GADDY_ZENODO_RECORD = "4064409"
GADDY_TAR_URL = "https://zenodo.org/records/4064409/files/emg_data.tar.gz"
GADDY_FS_HZ = 1000
GADDY_CHANNELS = 8
GADDY_TAR_BYTES = 3_919_507_637


def _download_with_progress(url: str, dest: Path, progress_cb: Optional[Callable[[str], None]] = None) -> None:
    dest = Path(dest)
    tmp = dest.with_suffix(dest.suffix + ".part")

    last_pct = [-1.0]

    def _hook(block_count: int, block_size: int, total_size: int) -> None:
        if progress_cb is None:
            return
        done = block_count * block_size
        total = total_size if total_size > 0 else GADDY_TAR_BYTES
        pct = min(100.0, 100.0 * done / max(total, 1))
        if pct - last_pct[0] < 1.0 and pct < 99.9:
            return
        last_pct[0] = pct
        progress_cb(f"download {pct:5.1f}% ({done / 1e6:.0f}/{total / 1e6:.0f} MB)")

    if progress_cb:
        progress_cb(f"downloading {url} (~{GADDY_TAR_BYTES / 1e9:.1f} GB)...")
    urllib.request.urlretrieve(url, tmp, reporthook=_hook)
    tmp.replace(dest)
    if progress_cb:
        progress_cb(f"download complete -> {dest}")


@dataclass
class GaddyImportReport:
    n_events: int
    n_skipped: int
    duration_s: float
    labels: List[str]
    out_dir: Path
    source: str = "gaddy_zenodo_4064409"

    def to_dict(self) -> dict:
        return {
            "n_events": int(self.n_events),
            "n_skipped": int(self.n_skipped),
            "duration_s": round(float(self.duration_s), 3),
            "labels": list(self.labels),
            "out_dir": str(self.out_dir),
            "source": self.source,
        }


def _slug_label(text: str, max_len: int = 48) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")
    if not s:
        return "unknown"
    return s[:max_len]


def _iter_gaddy_pairs(raw_dir: Path) -> Iterator[Tuple[Path, Path]]:
    raw_dir = Path(raw_dir)
    for emg_path in sorted(raw_dir.rglob("*_emg.npy")):
        stem = emg_path.name[: -len("_emg.npy")]
        info_path = emg_path.with_name(f"{stem}_info.json")
        if info_path.is_file():
            yield emg_path, info_path


def _load_gaddy_info(info_path: Path) -> dict:
    return json.loads(info_path.read_text(encoding="utf-8"))


def _label_from_info(info: dict, *, label_mode: str) -> Optional[str]:
    if int(info.get("sentence_index", 0)) < 0:
        return None
    text = str(info.get("text", "")).strip()
    if not text:
        return None
    mode = str(label_mode).lower()
    if mode == "sentence":
        return _slug_label(text)
    if mode == "first_word":
        parts = re.split(r"\s+", text)
        return _slug_label(parts[0]) if parts else None
    raise ValueError(f"unknown label_mode: {label_mode}")


def convert_gaddy_raw_dir(
    raw_dir: Path,
    out_dir: Path,
    *,
    silent_only: bool = True,
    label_mode: str = "first_word",
    max_samples: Optional[int] = None,
    gap_samples: int = 250,
    top_labels: Optional[int] = None,
    min_samples_per_label: int = 1,
) -> GaddyImportReport:
    """Convert extracted Gaddy raw files to an OpenAlterEgo session folder."""
    raw_dir = Path(raw_dir)
    segments: List[tuple[np.ndarray, str]] = []
    skipped = 0
    n_used = 0

    for emg_path, info_path in _iter_gaddy_pairs(raw_dir):
        if max_samples is not None and n_used >= int(max_samples):
            break
        info = _load_gaddy_info(info_path)
        if silent_only and not bool(info.get("silent", True)):
            skipped += 1
            continue
        label = _label_from_info(info, label_mode=label_mode)
        if label is None:
            skipped += 1
            continue
        emg = np.load(emg_path).astype(np.float32)
        if emg.ndim != 2 or emg.shape[1] != GADDY_CHANNELS:
            skipped += 1
            continue
        if emg.shape[0] < 32:
            skipped += 1
            continue
        segments.append((emg, str(label)))
        n_used += 1

    if not segments:
        raise ValueError(f"no usable Gaddy samples found under {raw_dir}")

    if int(min_samples_per_label) > 1 or top_labels is not None:
        from collections import Counter

        counts = Counter(lab for _, lab in segments)
        keep = [lab for lab, n in counts.items() if n >= int(min_samples_per_label)]
        if top_labels is not None:
            keep = sorted(keep, key=lambda lab: counts[lab], reverse=True)[: int(top_labels)]
        keep_set = set(keep)
        segments = [(emg, lab) for emg, lab in segments if lab in keep_set]

    chunks: List[np.ndarray] = []
    events: List[dict] = []
    offset = 0
    for emg, label in segments:
        start = offset + int(gap_samples)
        end = start + int(emg.shape[0])
        events.append({"start_sample": start, "end_sample": end, "label": label})
        chunks.append(emg)
        offset = end + int(gap_samples)

    signals = np.vstack(chunks).astype(np.float32)
    ev_df = pd.DataFrame(events)
    meta = SessionMeta(
        fs_hz=float(GADDY_FS_HZ),
        channels=int(GADDY_CHANNELS),
        source="gaddy_zenodo_4064409",
        notes=[
            "Gaddy & Klein EMNLP 2020 silent speech EMG",
            f"label_mode={label_mode}",
            f"silent_only={silent_only}",
        ],
        extra={"doi": "10.5281/zenodo.4064409"},
    )
    out = write_session_folder(out_dir, signals, ev_df, meta)
    labels = sorted(ev_df["label"].astype(str).unique().tolist())
    return GaddyImportReport(
        n_events=len(ev_df),
        n_skipped=int(skipped),
        duration_s=float(signals.shape[0]) / float(GADDY_FS_HZ),
        labels=labels,
        out_dir=out,
    )


def download_gaddy_archive(
    dest_dir: Path,
    *,
    max_members: Optional[int] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Path:
    """Download Gaddy ``emg_data.tar.gz`` and optionally extract a subset of members.

    The full archive is ~3.9 GB. When ``max_members`` is set, only the first N
    ``*_emg.npy`` / ``*_info.json`` pairs are extracted (still requires full download).
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    tar_path = dest_dir / "emg_data.tar.gz"
    raw_dir = dest_dir / "raw"

    if not tar_path.is_file():
        _download_with_progress(GADDY_TAR_URL, tar_path, progress_cb=progress_cb)

    raw_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tf:
        members = [m for m in tf.getmembers() if m.name.endswith("_emg.npy") or m.name.endswith("_info.json")]
        if max_members is not None:
            stems: List[str] = []
            for m in members:
                if not m.name.endswith("_emg.npy"):
                    continue
                stem = Path(m.name).name[: -len("_emg.npy")]
                if stem not in stems:
                    if len(stems) >= int(max_members):
                        break
                    stems.append(stem)
            stem_set = set(stems)
            members = [
                m
                for m in members
                if (Path(m.name).name[: -len("_emg.npy")] in stem_set and m.name.endswith("_emg.npy"))
                or (Path(m.name).name[: -len("_info.json")] in stem_set and m.name.endswith("_info.json"))
            ]
        if progress_cb:
            progress_cb(f"extracting {len(members)} files from archive...")
        for member in members:
            tf.extract(member, path=raw_dir)
    if progress_cb:
        progress_cb(f"raw EMG extracted -> {raw_dir}")
    return raw_dir


def import_gaddy_session(
    out_dir: Path,
    *,
    raw_dir: Optional[Path] = None,
    download_dir: Optional[Path] = None,
    max_samples: Optional[int] = 40,
    silent_only: bool = True,
    label_mode: str = "first_word",
    skip_download: bool = False,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> GaddyImportReport:
    """Download (optional) + convert Gaddy data into ``out_dir`` session format."""
    raw: Optional[Path] = Path(raw_dir) if raw_dir else None
    if raw is None or not any(raw.rglob("*_emg.npy")):
        if skip_download:
            raise FileNotFoundError(
                f"Gaddy raw EMG not found under {raw_dir}; omit --skip-download to fetch Zenodo archive"
            )
        dl_root = Path(download_dir) if download_dir else Path(out_dir).parent / "gaddy_download"
        raw = download_gaddy_archive(
            dl_root,
            max_members=max_samples,
            progress_cb=progress_cb,
        )
    return convert_gaddy_raw_dir(
        raw,
        out_dir,
        silent_only=silent_only,
        label_mode=label_mode,
        max_samples=max_samples,
    )
