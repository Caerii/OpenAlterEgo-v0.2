"""Reproducible, nested sampling of unique complete measurement programs."""

from collections.abc import Sequence

import numpy as np


def uniform_program_prefix(options: Sequence[Sequence[tuple[int, ...]]],
                           count: int, seed: int) -> np.ndarray:
    """Draw complete grammar programs in encounter order, preserving prefixes.

    This is uniform over the Cartesian product before duplicate rejection.
    Grammars with multiple representations of one physical program would not
    produce a uniform distribution over unique physical programs.
    """
    if count < 1 or not options or any(not group for group in options):
        raise ValueError("nonempty grammar and positive count required")
    rng = np.random.default_rng(seed)
    seen = set()
    programs = []
    attempts = 0
    while len(programs) < count:
        attempts += 1
        if attempts > count * 1000:
            raise ValueError("not enough unique feasible programs")
        choice = tuple(sorted(channel for group in options
                              for channel in group[rng.integers(len(group))]))
        if len(choice) != len(set(choice)) or choice in seen:
            continue
        seen.add(choice)
        programs.append(choice)
    return np.asarray(programs, dtype=np.int16)
