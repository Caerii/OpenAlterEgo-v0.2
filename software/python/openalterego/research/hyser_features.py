"""Publisher-format Hyser WFDB-16 reader and trial-level EMG features."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np


GAIN = re.compile(r"(?P<gain>[0-9.]+)\((?P<baseline>-?[0-9]+)\)/V")
SITE = re.compile(r"(?P<grid>[A-Z]{2})-(?P<row>[1-8])-(?P<column>[1-8])")


def read_wfdb_16(base: Path) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return physical-valued samples after verifying WFDB initial/checksums."""
    header = base.with_suffix(".hea").read_text(encoding="ascii").splitlines()
    first = header[0].split()
    if len(first) < 4 or first[0] != base.name or int(first[1]) != 256 or int(first[2]) != 2048:
        raise ValueError(f"unexpected Hyser record header: {base}")
    count = int(first[3])
    if count < 2 or len(header) != 257:
        raise ValueError(f"unexpected Hyser channel/sample count: {base}")
    gain, baseline, initial, checksum, sites = [], [], [], [], []
    for line in header[1:]:
        fields = line.split()
        if (len(fields) < 9 or fields[0] != base.name + ".dat" or
                fields[1] != "16"):
            raise ValueError(f"unexpected Hyser signal specification: {base}")
        match = GAIN.fullmatch(fields[2])
        if match is None or SITE.fullmatch(fields[-1]) is None:
            raise ValueError(f"unexpected gain or grid site: {base}")
        gain.append(float(match["gain"]))
        baseline.append(int(match["baseline"]))
        initial.append(int(fields[5]))
        checksum.append(int(fields[6]))
        sites.append(fields[-1])
    if len(set(sites)) != 256 or any(value <= 0 for value in gain):
        raise ValueError(f"invalid channel geometry or gain: {base}")
    raw = np.fromfile(base.with_suffix(".dat"), dtype="<i2")
    if raw.size != count * 256:
        raise ValueError(f"signal length mismatch: {base}")
    raw = raw.reshape(count, 256)
    if (not np.array_equal(raw[0], initial) or
            not np.array_equal(raw.astype(np.int64).sum(axis=0) % 65536,
                               np.asarray(checksum) % 65536)):
        raise ValueError(f"WFDB initial/checksum mismatch: {base}")
    signal = (raw.astype(np.float32) - np.asarray(baseline, dtype=np.float32)) / np.asarray(
        gain, dtype=np.float32)
    if not np.isfinite(signal).all():
        raise ValueError(f"nonfinite physical signal: {base}")
    return signal, tuple(sites)


def trial_features(signal: np.ndarray) -> np.ndarray:
    """Whole-trial log RMS and log waveform length, interleaved by channel."""
    if signal.ndim != 2 or signal.shape[1] != 256 or len(signal) < 2:
        raise ValueError("expected 256-channel time series")
    rms = np.sqrt(np.mean(np.square(signal, dtype=np.float64), axis=0))
    waveform = np.mean(np.abs(np.diff(signal.astype(np.float64), axis=0)), axis=0)
    return np.stack((np.log(np.maximum(rms, 1e-8)),
                     np.log(np.maximum(waveform, 1e-8))), axis=1).reshape(-1).astype(np.float32)


def site_coordinates(sites: tuple[str, ...]) -> np.ndarray:
    """Grid, row, column labels; geometry is topological, not millimetres."""
    if len(sites) != 256 or len(set(sites)) != 256:
        raise ValueError("expected 256 distinct sites")
    values = []
    grid_ids = {name: index for index, name in enumerate(sorted({site[:2] for site in sites}))}
    if len(grid_ids) != 4:
        raise ValueError("expected four 8x8 grids")
    for site in sites:
        match = SITE.fullmatch(site)
        if match is None:
            raise ValueError("unexpected site label")
        values.append((grid_ids[match["grid"]], int(match["row"]) - 1,
                       int(match["column"]) - 1))
    return np.asarray(values, dtype=np.int8)
