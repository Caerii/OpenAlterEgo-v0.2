"""Recorded-grid displacement stress test for candidate electrode layouts."""

import numpy as np

from openalterego.research.hyser_search import feasible, score_designs


OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def displacement_maps(sites: np.ndarray) -> list[np.ndarray]:
    if sites.shape != (256, 3):
        raise ValueError("expected four complete 8x8 grids")
    lookup = {tuple(map(int, location)): index for index, location in enumerate(sites)}
    if len(lookup) != 256:
        raise ValueError("duplicate sites")
    return [np.asarray([lookup.get((int(grid), int(row) + dr, int(col) + dc), -1)
                        for grid, row, col in sites], dtype=np.int16)
            for dr, dc in OFFSETS]


def candidate_designs(sites: np.ndarray, *, count: int, seed: int = 42,
                      max_per_grid: int | None = 2) -> np.ndarray:
    if count < 1:
        raise ValueError("count must be positive")
    eligible = np.flatnonzero((sites[:, 1] >= 1) & (sites[:, 1] <= 6) &
                              (sites[:, 2] >= 1) & (sites[:, 2] <= 6))
    rng = np.random.default_rng(seed)
    designs = set()
    while len(designs) < count:
        design = tuple(sorted(map(int, rng.choice(eligible, size=8, replace=False))))
        if feasible(design, sites, max_per_grid=max_per_grid):
            designs.add(design)
    return np.asarray(sorted(designs), dtype=np.int16)


def displacement_scores(folds, designs: np.ndarray, maps: list[np.ndarray]) -> np.ndarray:
    """CV score at nominal sites and four one-cell translated layouts."""
    scores = [score_designs(folds, designs)]
    for shift in maps:
        moved = shift[designs]
        if np.any(moved < 0) or any(len(set(row)) != len(row) for row in moved):
            raise ValueError("design is not shiftable on its recorded grid")
        scores.append(score_designs(folds, moved))
    return np.stack(scores, axis=1)
