"""Reproducible, nested sampling of unique complete measurement programs."""

from collections.abc import Sequence

import numpy as np


def _uniform_program_prefix(options: Sequence[Sequence[tuple[int, ...]]],
                            count: int, seed: int,
                            *, with_codes: bool) -> tuple[np.ndarray, np.ndarray | None]:
    """Draw complete programs and optionally retain their grammar choices.

    This is uniform over the Cartesian product before duplicate rejection.
    Grammars with multiple representations of one physical program would not
    produce a uniform distribution over unique physical programs.
    """
    if count < 1 or not options or any(not group for group in options):
        raise ValueError("nonempty grammar and positive count required")
    rng = np.random.default_rng(seed)
    seen = set()
    programs = []
    codes = [] if with_codes else None
    attempts = 0
    while len(programs) < count:
        attempts += 1
        if attempts > count * 1000:
            raise ValueError("not enough unique feasible programs")
        indices = tuple(int(rng.integers(len(group))) for group in options)
        choice = tuple(sorted(channel for group, index in zip(options, indices)
                              for channel in group[index]))
        if len(choice) != len(set(choice)) or choice in seen:
            continue
        seen.add(choice)
        programs.append(choice)
        if codes is not None:
            codes.append(indices)
    return (np.asarray(programs, dtype=np.int16),
            np.asarray(codes, dtype=np.int16) if codes is not None else None)


def uniform_program_prefix(options: Sequence[Sequence[tuple[int, ...]]],
                           count: int, seed: int) -> np.ndarray:
    """Draw complete grammar programs in encounter order, preserving prefixes."""
    return _uniform_program_prefix(options, count, seed, with_codes=False)[0]


def uniform_program_prefix_with_codes(
        options: Sequence[Sequence[tuple[int, ...]]], count: int,
        seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Also return each sampled program's option index in every grammar part."""
    programs, codes = _uniform_program_prefix(options, count, seed,
                                             with_codes=True)
    assert codes is not None
    return programs, codes
