"""Combinatorial smoothing over one-channel substitutions of recorded montages."""

from __future__ import annotations

from itertools import combinations

import numpy as np


def one_swap_smoothed_scores(
    programs: list[tuple[str, tuple[int, ...]]], scores: np.ndarray,
    *, n_channels: int = 14, width: int = 6,
) -> np.ndarray:
    """Average each score with every feasible one-channel-substitution neighbor.

    The full subset lattice must be supplied for each normalization. Channel
    index is merely an identity here; this is not a spatial electrode model.
    """
    scores = np.asarray(scores, dtype=float)
    norms = {normalization for normalization, _ in programs}
    expected = len(norms) * len(list(combinations(range(n_channels), width)))
    if (scores.shape != (len(programs),) or len(programs) != expected or
            len(set(programs)) != len(programs) or not np.isfinite(scores).all() or
            not 0 < width < n_channels or
            any(len(channels) != width or tuple(sorted(set(channels))) != channels or
                channels[0] < 0 or channels[-1] >= n_channels
                for _, channels in programs)):
        raise ValueError("expected a complete, finite channel-subset score lattice")
    lookup = {program: index for index, program in enumerate(programs)}
    smoothed = np.empty_like(scores)
    for index, (normalization, channels) in enumerate(programs):
        chosen = set(channels)
        neighbors = [lookup[(normalization, tuple(sorted(chosen - {old} | {new})))]
                     for old in channels for new in range(n_channels) if new not in chosen]
        smoothed[index] = (scores[index] + scores[neighbors].sum()) / (1 + len(neighbors))
    return smoothed
