"""Day-one grid-neighbor consistency rule for recorded Hyser contacts."""

import numpy as np

from openalterego.research.hyser_search import feasible, score_designs


def dark_contact_mask(features: np.ndarray, sites: np.ndarray,
                      *, log_rms_gap: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Veto contacts whose median log RMS is much below adjacent grid sites.

    This is an observational quality heuristic, not an electrode-fault label.
    Coordinates are four separate 8x8 grids and do not imply millimetres.
    """
    if features.ndim != 2 or features.shape[1] != 512 or sites.shape != (256, 3):
        raise ValueError("expected Hyser trial features and grid coordinates")
    if not np.isfinite(features).all() or log_rms_gap <= 0:
        raise ValueError("invalid site-health input")
    medians = np.median(features[:, ::2], axis=0)
    darkness = np.empty(256, dtype=np.float64)
    for index, (grid, row, col) in enumerate(sites):
        adjacent = np.flatnonzero((sites[:, 0] == grid) &
                                  (np.abs(sites[:, 1] - row) +
                                   np.abs(sites[:, 2] - col) == 1))
        if not len(adjacent):
            raise ValueError("site has no adjacent grid contacts")
        darkness[index] = max(0.0, float(np.median(medians[adjacent]) - medians[index]))
    return darkness <= log_rms_gap, darkness


def screened_greedy(folds, sites: np.ndarray, allowed: np.ndarray,
                    *, width: int = 8, max_per_grid: int | None = None):
    if allowed.shape != (256,) or allowed.dtype != np.bool_:
        raise ValueError("expected 256-site boolean mask")
    chosen = []
    evaluated = 0
    for _ in range(width):
        candidates = [index for index in np.flatnonzero(allowed)
                      if feasible(tuple(chosen + [int(index)]), sites,
                                  max_per_grid=max_per_grid)]
        if not candidates:
            raise ValueError("not enough feasible sites after health screen")
        designs = np.asarray([chosen + [int(index)] for index in candidates], dtype=np.int16)
        scores = score_designs(folds, designs)
        evaluated += len(candidates)
        best = min(range(len(candidates)), key=lambda i: (-scores[i], candidates[i]))
        chosen.append(int(candidates[best]))
    return chosen, float(scores[best]), evaluated
