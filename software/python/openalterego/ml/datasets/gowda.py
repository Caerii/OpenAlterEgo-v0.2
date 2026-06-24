"""Gowda orofacial EMG import adapters (OSF YM5JD / emg2speech small-vocab).

Expected small-vocab files (manual download from OSF):
  https://osf.io/ym5jd/  or  https://osf.io/bgh7t/files/box
  dataSmallVocab.npy   — EMG segments (object array or 3D float array)
  labelsSmallVocab.npy — parallel labels (strings or ints)

Also supports per-subject NATO word folders:
  Subject 1/ ... raw trial files (``.npy`` blocks with timestamps in sidecar JSON if present)
"""

from __future__ import annotations

import json
import pickle
import re
import shutil
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .session import SessionMeta, write_session_folder

GOWDA_SMALL_VOCAB_FS_HZ = 5000.0
GOWDA_TRIAL_SAMPLES = 45000
# Official emg2speech small-vocab word boundaries (2 s × 3 + 3 s), not equal quarters.
GOWDA_WORD_SLICE_SAMPLES: Tuple[int, ...] = (10000, 10000, 10000, 15000)
GOWDA_OSF_DOI = "10.17605/OSF.IO/YM5JD"
GOWDA_OSF_PROJECT = "bgh7t"
GOWDA_DATA_URL = "https://osf.io/download/cj9kb/"  # dataSmallVocab.npy (emg2speech box)
GOWDA_LABELS_URL = "https://osf.io/download/htpcg/"  # labelsSmallVocab.npy


@dataclass
class GowdaImportReport:
    n_events: int
    duration_s: float
    labels: List[str]
    channels: int
    fs_hz: float
    out_dir: Path
    source: str = "gowda_small_vocab"

    def to_dict(self) -> dict:
        return {
            "n_events": int(self.n_events),
            "duration_s": round(float(self.duration_s), 3),
            "labels": list(self.labels),
            "channels": int(self.channels),
            "fs_hz": float(self.fs_hz),
            "out_dir": str(self.out_dir),
            "source": self.source,
        }


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")
    return s or "unknown"


def _finalize_download(tmp: Path, dest: Path) -> None:
    """Move completed ``.part`` download into place (Windows may lock briefly)."""
    if dest.is_file() and dest.stat().st_size == tmp.stat().st_size:
        tmp.unlink(missing_ok=True)
        return
    last_err: Optional[BaseException] = None
    for attempt in range(8):
        try:
            if dest.is_file():
                dest.unlink()
            shutil.move(str(tmp), str(dest))
            return
        except (PermissionError, OSError) as err:
            last_err = err
            time.sleep(0.25 * (attempt + 1))
    if last_err is not None:
        raise last_err


def _download_osf_file(url: str, dest: Path, progress_cb: Optional[Callable[[str], None]] = None) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    if dest.is_file():
        return dest
    if tmp.is_file() and not dest.is_file():
        _finalize_download(tmp, dest)
        return dest

    last_pct = [-1.0]

    def _hook(block_count: int, block_size: int, total_size: int) -> None:
        if progress_cb is None:
            return
        done = block_count * block_size
        total = total_size if total_size > 0 else max(done, 1)
        pct = min(100.0, 100.0 * done / total)
        if pct - last_pct[0] < 1.0 and pct < 99.9:
            return
        last_pct[0] = pct
        progress_cb(f"download {dest.name}: {pct:5.1f}% ({done / 1e6:.0f}/{total / 1e6:.0f} MB)")

    if progress_cb:
        progress_cb(f"downloading {url} -> {dest.name}")
    urllib.request.urlretrieve(url, tmp, reporthook=_hook)
    _finalize_download(tmp, dest)
    return dest


def download_gowda_small_vocab(
    dest_dir: Path,
    *,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Tuple[Path, Path]:
    """Download Gowda small-vocab npy pair from OSF emg2speech project (bgh7t)."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    data_path = dest_dir / "dataSmallVocab.npy"
    labels_path = dest_dir / "labelsSmallVocab.npy"
    if not data_path.is_file():
        _download_osf_file(GOWDA_DATA_URL, data_path, progress_cb=progress_cb)
    if not labels_path.is_file():
        _download_osf_file(GOWDA_LABELS_URL, labels_path, progress_cb=progress_cb)
    return data_path, labels_path


def import_gowda_small_vocab_from_osf(
    out_dir: Path,
    *,
    download_dir: Optional[Path] = None,
    fs_hz: float = GOWDA_SMALL_VOCAB_FS_HZ,
    max_segments: Optional[int] = None,
    top_labels: Optional[int] = None,
    min_samples_per_label: int = 1,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> GowdaImportReport:
    """Download + convert Gowda small-vocab in one step."""
    dl = Path(download_dir) if download_dir else Path(out_dir).parent / "gowda_download"
    data_path, labels_path = download_gowda_small_vocab(dl, progress_cb=progress_cb)
    return import_gowda_small_vocab(
        data_path,
        labels_path,
        out_dir,
        fs_hz=float(fs_hz),
        max_segments=max_segments,
        top_labels=top_labels,
        min_samples_per_label=min_samples_per_label,
    )


def _normalize_segments(data: Any) -> List[np.ndarray]:
    """Coerce Gowda small-vocab arrays into a list of (time, channels) float32 segments."""
    if isinstance(data, np.ndarray):
        if data.ndim == 3:
            return [data[i].astype(np.float32) for i in range(int(data.shape[0]))]
        if data.dtype == object:
            if data.ndim == 1:
                return [np.asarray(x, dtype=np.float32) for x in data.tolist()]
            raise ValueError(f"object EMG array must be 1D of segments, got shape {data.shape}")
        if data.ndim == 2:
            return [data.astype(np.float32)]
    if isinstance(data, list):
        return [np.asarray(x, dtype=np.float32) for x in data]
    raise ValueError(f"unsupported Gowda data layout: type={type(data)} shape={getattr(data, 'shape', None)}")


def _normalize_labels(labels: Any, n: int) -> List[str]:
    if isinstance(labels, np.ndarray):
        labels = labels.ravel().tolist()
    if isinstance(labels, list):
        out = [_slug(str(x)) for x in labels]
    else:
        raise ValueError(f"unsupported labels type: {type(labels)}")
    if len(out) != n:
        raise ValueError(f"label count {len(out)} != segment count {n}")
    return out


def _load_labels_file(labels_path: Path) -> Any:
    path = Path(labels_path)
    if path.suffix.lower() in (".pkl", ".pickle"):
        with path.open("rb") as f:
            return pickle.load(f)
    return np.load(path, allow_pickle=True)


def _load_data_file(data_path: Path) -> Any:
    path = Path(data_path)
    if path.suffix.lower() in (".pkl", ".pickle"):
        with path.open("rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict):
            for key in ("data", "emg", "X", "segments"):
                if key in obj:
                    return obj[key]
        return obj
    return np.load(path, allow_pickle=True)


def _zscore_trial_channels(trial_ch_time: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-trial z-score along time per channel (matches emg2speech smallVocab notebook)."""
    x = np.asarray(trial_ch_time, dtype=np.float32)
    mu = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    return (x - mu) / (std + float(eps))


def _gowda_word_sample_ranges(trial_samples: int, n_words: int) -> List[Tuple[int, int]]:
    """Return ``(start, end)`` sample ranges for each word in a trial."""
    n_words = int(n_words)
    if (
        int(trial_samples) == GOWDA_TRIAL_SAMPLES
        and n_words == len(GOWDA_WORD_SLICE_SAMPLES)
    ):
        bounds = [0]
        for length in GOWDA_WORD_SLICE_SAMPLES:
            bounds.append(bounds[-1] + int(length))
        return [(bounds[w], bounds[w + 1]) for w in range(n_words)]
    seg_len = max(1, int(trial_samples) // n_words)
    return [
        (w * seg_len, int(trial_samples) if w == n_words - 1 else (w + 1) * seg_len)
        for w in range(n_words)
    ]


def _is_gowda_trial_cube(data: np.ndarray) -> bool:
    """True for OSF layout ``(trials, channels, time)`` with channels << time."""
    if not isinstance(data, np.ndarray) or data.ndim != 3:
        return False
    _trials, channels, time = data.shape
    return int(channels) < int(time) and int(channels) >= 8


def _import_gowda_trial_cube(
    data: np.ndarray,
    labels: np.ndarray,
    out_dir: Path,
    *,
    fs_hz: float,
    max_trials: Optional[int] = None,
    top_labels: Optional[int] = None,
    min_samples_per_label: int = 1,
    words_per_trial: int = 4,
    zscore_trials: bool = True,
) -> GowdaImportReport:
    """Import Gowda ``(trials, channels, time)`` + ``(trials, words)`` label grid."""
    n_trials = min(int(data.shape[0]), int(labels.shape[0]))
    if max_trials is not None:
        n_trials = min(n_trials, int(max_trials))

    # Collect word-level events first for label filtering
    pending: List[tuple[np.ndarray, str, int, int]] = []
    for i in range(n_trials):
        trial = np.asarray(data[i], dtype=np.float32)  # (ch, time)
        if trial.ndim != 2:
            continue
        if zscore_trials:
            trial = _zscore_trial_channels(trial)
        trial = trial.T  # (time, ch)
        n_words = min(int(words_per_trial), int(labels.shape[1]))
        for w, (s0, s1) in enumerate(_gowda_word_sample_ranges(int(trial.shape[0]), n_words)):
            word = _slug(str(labels[i, w]))
            pending.append((trial[s0:s1, :], word, int(i), int(w)))

    if int(min_samples_per_label) > 1 or top_labels is not None:
        from collections import Counter

        counts = Counter(lab for _, lab, _, _ in pending)
        keep = [lab for lab, n in counts.items() if n >= int(min_samples_per_label)]
        if top_labels is not None:
            keep = sorted(keep, key=lambda lab: counts[lab], reverse=True)[: int(top_labels)]
        keep_set = set(keep)
        pending = [(seg, lab, tid, wid) for seg, lab, tid, wid in pending if lab in keep_set]

    chunks: List[np.ndarray] = []
    events: List[dict] = []
    offset = 0
    ch_count: Optional[int] = None
    for seg, lab, trial_id, word_idx in pending:
        if seg.shape[0] < 16:
            continue
        ch_count = int(seg.shape[1])
        start = offset
        end = start + int(seg.shape[0])
        events.append(
            {
                "start_sample": start,
                "end_sample": end,
                "label": lab,
                "trial_id": int(trial_id),
                "word_idx": int(word_idx),
            }
        )
        chunks.append(seg.astype(np.float32, copy=False))
        offset = end

    if not chunks or ch_count is None:
        raise ValueError("no usable Gowda word segments after filtering")

    signals = np.vstack(chunks).astype(np.float32)
    ev_df = pd.DataFrame(events)
    meta = SessionMeta(
        fs_hz=float(fs_hz),
        channels=int(ch_count),
        source="gowda_small_vocab",
        notes=[
            "Gowda small-vocab: official 10k/10k/10k/15k word slices + per-trial z-score",
            f"trials={n_trials} words_per_trial={words_per_trial}",
        ],
        extra={"doi": GOWDA_OSF_DOI},
    )
    out = write_session_folder(out_dir, signals, ev_df, meta)
    uniq = sorted(ev_df["label"].astype(str).unique().tolist())
    return GowdaImportReport(
        n_events=len(ev_df),
        duration_s=float(signals.shape[0]) / float(fs_hz),
        labels=uniq,
        channels=int(ch_count),
        fs_hz=float(fs_hz),
        out_dir=Path(out_dir),
    )


def import_gowda_small_vocab(
    data_path: Union[str, Path],
    labels_path: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    fs_hz: float = GOWDA_SMALL_VOCAB_FS_HZ,
    max_segments: Optional[int] = None,
    top_labels: Optional[int] = None,
    min_samples_per_label: int = 1,
    channel_slice: Optional[slice] = None,
) -> GowdaImportReport:
    """Convert Gowda ``dataSmallVocab`` + ``labelsSmallVocab`` to a session folder."""
    data = _load_data_file(Path(data_path))
    labels_raw = _load_labels_file(Path(labels_path))

    if isinstance(data, np.ndarray) and _is_gowda_trial_cube(data):
        labels_arr = np.asarray(labels_raw)
        if labels_arr.ndim != 2:
            raise ValueError(f"expected 2D word label grid, got shape {labels_arr.shape}")
        return _import_gowda_trial_cube(
            data,
            labels_arr,
            Path(out_dir),
            fs_hz=float(fs_hz),
            max_trials=max_segments,
            top_labels=top_labels,
            min_samples_per_label=int(min_samples_per_label),
        )

    segments = _normalize_segments(data)
    labels = _normalize_labels(labels_raw, len(segments))
    paired = list(zip(segments, labels))
    if max_segments is not None:
        paired = paired[: int(max_segments)]

    if int(min_samples_per_label) > 1 or top_labels is not None:
        from collections import Counter

        counts = Counter(lab for _, lab in paired)
        keep = [lab for lab, n in counts.items() if n >= int(min_samples_per_label)]
        if top_labels is not None:
            keep = sorted(keep, key=lambda lab: counts[lab], reverse=True)[: int(top_labels)]
        keep_set = set(keep)
        paired = [(seg, lab) for seg, lab in paired if lab in keep_set]

    chunks: List[np.ndarray] = []
    events: List[dict] = []
    offset = 0
    ch_count: Optional[int] = None

    for seg, lab in paired:
        x = np.asarray(seg, dtype=np.float32)
        if x.ndim != 2:
            continue
        if channel_slice is not None:
            x = x[:, channel_slice]
        if x.shape[0] < 16:
            continue
        ch_count = int(x.shape[1])
        start = offset
        end = start + int(x.shape[0])
        events.append({"start_sample": start, "end_sample": end, "label": lab})
        chunks.append(x)
        offset = end

    if not chunks or ch_count is None:
        raise ValueError("no usable Gowda segments after filtering")

    signals = np.vstack(chunks).astype(np.float32)
    ev_df = pd.DataFrame(events)
    meta = SessionMeta(
        fs_hz=float(fs_hz),
        channels=int(ch_count),
        source="gowda_small_vocab",
        notes=[
            "Gowda et al. 2024/2025 orofacial EMG small vocabulary",
            f"data={Path(data_path).name}",
        ],
        extra={"doi": GOWDA_OSF_DOI},
    )
    out = write_session_folder(out_dir, signals, ev_df, meta)
    uniq = sorted(ev_df["label"].astype(str).unique().tolist())
    return GowdaImportReport(
        n_events=len(ev_df),
        duration_s=float(signals.shape[0]) / float(fs_hz),
        labels=uniq,
        channels=int(ch_count),
        fs_hz=float(fs_hz),
        out_dir=Path(out_dir),
    )


def import_gowda_nato_subject(
    subject_dir: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    fs_hz: float = GOWDA_SMALL_VOCAB_FS_HZ,
    label_from: str = "filename",
) -> GowdaImportReport:
    """Import a Gowda ``Subject N`` NATO-words folder of ``.npy`` trial files."""
    subject_dir = Path(subject_dir)
    npy_files = sorted(subject_dir.glob("*.npy"))
    if not npy_files:
        npy_files = sorted(subject_dir.rglob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"no .npy trials under {subject_dir}")

    chunks: List[np.ndarray] = []
    events: List[dict] = []
    offset = 0
    ch_count: Optional[int] = None

    for f in npy_files:
        x = np.load(f).astype(np.float32)
        if x.ndim != 2 or x.shape[0] < 16:
            continue
        if label_from == "filename":
            lab = _slug(f.stem)
        else:
            lab = _slug(f.stem)
        ch_count = int(x.shape[1])
        start = offset
        end = start + int(x.shape[0])
        events.append({"start_sample": start, "end_sample": end, "label": lab})
        chunks.append(x)
        offset = end

    if not chunks or ch_count is None:
        raise ValueError(f"no usable trials in {subject_dir}")

    signals = np.vstack(chunks).astype(np.float32)
    ev_df = pd.DataFrame(events)
    meta = SessionMeta(
        fs_hz=float(fs_hz),
        channels=int(ch_count),
        source="gowda_nato_words",
        notes=[f"subject_dir={subject_dir.name}"],
        extra={"doi": GOWDA_OSF_DOI},
    )
    out = write_session_folder(out_dir, signals, ev_df, meta)
    uniq = sorted(ev_df["label"].astype(str).unique().tolist())
    return GowdaImportReport(
        n_events=len(ev_df),
        duration_s=float(signals.shape[0]) / float(fs_hz),
        labels=uniq,
        channels=int(ch_count),
        fs_hz=float(fs_hz),
        out_dir=Path(out_dir),
    )


def write_dataset_catalog(path: Union[str, Path]) -> Path:
    """Write a JSON catalog of known public EMG datasets and import commands."""
    catalog = {
        "datasets": [
            {
                "id": "gaddy_silent_speech",
                "title": "Gaddy Silent Speech EMG (EMNLP 2020)",
                "url": "https://doi.org/10.5281/zenodo.4064409",
                "size_gb": 3.9,
                "fs_hz": 1000,
                "channels": 8,
                "license": "CC-BY-4.0",
                "import": "openalterego dataset import gaddy --out ./sessions/gaddy",
            },
            {
                "id": "gowda_geometry",
                "title": "Gowda Orofacial EMG (Geometry paper, OSF YM5JD)",
                "url": "https://doi.org/10.17605/OSF.IO/YM5JD",
                "size_gb": 5.5,
                "fs_hz": 5000,
                "channels": 31,
                "license": "see OSF",
                "import": "openalterego dataset import-gowda --download --out ./sessions/gowda_sv",
                "manual_download": "https://osf.io/bgh7t/files/box (small-vocab npy files)",
            },
        ]
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    return out
