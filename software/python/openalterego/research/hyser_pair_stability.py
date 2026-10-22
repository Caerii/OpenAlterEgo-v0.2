"""Donor-session drift prior for measured adjacent-pair EMG programs."""

import numpy as np


def pair_drift_ranks(day1: tuple[np.ndarray, np.ndarray],
                     day2: tuple[np.ndarray, np.ndarray],
                     pairs: list[list[int]],
                     sites: list[str]) -> np.ndarray:
    """Within-grid rank of class-conditional day shift relative to class signal."""
    x1, y1 = day1
    x2, y2 = day2
    count = len(pairs)
    if (x1.ndim != 2 or x2.ndim != 2 or
            x1.shape[1] != 2 * count or x2.shape[1] != 2 * count or
            len(y1) != len(x1) or len(y2) != len(x2) or
            not np.isfinite(x1).all() or not np.isfinite(x2).all() or
            len(sites) != 256 or count < 1):
        raise ValueError("invalid paired feature arrays")
    classes = np.intersect1d(y1, y2)
    if len(classes) < 2:
        raise ValueError("no repeated classes across sessions")
    scale = np.maximum(x1.std(axis=0), 0.1)
    means1 = np.stack([x1[y1 == label].mean(axis=0) for label in classes])
    means2 = np.stack([x2[y2 == label].mean(axis=0) for label in classes])
    drift = np.mean(((means2 - means1) / scale) ** 2,
                    axis=0).reshape(count, 2).mean(axis=1)
    signal = np.var(means1 / scale, axis=0).reshape(count, 2).mean(axis=1)
    ratio = drift / (signal + 0.5)
    grids = np.asarray([sites[left][:2] for left, right in pairs])
    if any(sites[left][:2] != sites[right][:2] for left, right in pairs):
        raise ValueError("a pair crosses grids")
    ranks = np.empty(count, dtype=np.float64)
    for grid in np.unique(grids):
        ids = np.flatnonzero(grids == grid)
        order = np.argsort(ratio[ids], kind="stable")
        ranks[ids[order]] = np.arange(len(ids)) / max(1, len(ids) - 1)
    return ranks


def rank_programs(cv: np.ndarray, designs: np.ndarray,
                  donor_ranks: np.ndarray, penalty: float) -> np.ndarray:
    """Higher is better; penalty strength is in balanced-accuracy units."""
    cv = np.asarray(cv, dtype=np.float64)
    designs = np.asarray(designs, dtype=np.int16)
    donor_ranks = np.asarray(donor_ranks, dtype=np.float64)
    if (designs.ndim != 2 or cv.shape != (len(designs),) or
            donor_ranks.ndim != 1 or np.any(designs < 0) or
            np.any(designs >= len(donor_ranks)) or
            not np.isfinite(cv).all() or not np.isfinite(donor_ranks).all() or
            not np.isfinite(penalty) or penalty < 0):
        raise ValueError("invalid program ranking inputs")
    return cv - penalty * donor_ranks[designs].mean(axis=1)
