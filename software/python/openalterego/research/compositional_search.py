"""Batched beam and matched-count random search over discrete program parts."""

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class ProgramSearchResult:
    channels: tuple[int, ...]
    score: float
    evaluations: int
    stage_evaluations: tuple[int, ...]


def _validate_options(options: Sequence[Sequence[tuple[int, ...]]]):
    if not options or any(not group for group in options):
        raise ValueError("every program part needs options")
    for group in options:
        widths = {len(option) for option in group}
        if len(widths) != 1 or next(iter(widths)) < 1:
            raise ValueError("inconsistent option width")
        for option in group:
            if (len(set(option)) != len(option) or
                    any(channel < 0 or channel > np.iinfo(np.int16).max
                        for channel in option)):
                raise ValueError("invalid channel option")


def beam_search(options: Sequence[Sequence[tuple[int, ...]]],
                score: Callable[[np.ndarray], np.ndarray], *,
                beam_size: int = 16,
                feasible: Callable[[tuple[int, ...]], bool] | None = None,
                can_complete: Callable[[tuple[int, ...], int], bool] | None = None
                ) -> ProgramSearchResult:
    """Keep highest-scoring partial programs after adding each part."""
    _validate_options(options)
    if beam_size < 1:
        raise ValueError("beam size must be positive")
    beam = [tuple()]
    stage_counts = []
    final_scores = None
    for stage, group in enumerate(options):
        remaining = len(options) - stage - 1
        candidates_set = set()
        for prefix in beam:
            used = set(prefix)
            for option in group:
                if used.intersection(option):
                    continue
                candidate = tuple(sorted(prefix + option))
                if (feasible is not None and not feasible(candidate)) or (
                        can_complete is not None and not can_complete(
                            candidate, remaining)):
                    continue
                candidates_set.add(candidate)
        candidates = sorted(candidates_set)
        if not candidates:
            raise ValueError("no compatible full program")
        scores = np.asarray(score(np.asarray(candidates, dtype=np.int16)),
                            dtype=np.float64)
        if scores.shape != (len(candidates),) or not np.isfinite(scores).all():
            raise ValueError("scorer returned invalid shape or values")
        ranked = sorted(range(len(candidates)),
                        key=lambda i: (-scores[i], candidates[i]))
        beam = [candidates[i] for i in ranked[:beam_size]]
        final_scores = [float(scores[i]) for i in ranked[:beam_size]]
        stage_counts.append(len(candidates))
    return ProgramSearchResult(beam[0], final_scores[0],
                               sum(stage_counts), tuple(stage_counts))


def random_search(options: Sequence[Sequence[tuple[int, ...]]],
                  score: Callable[[np.ndarray], np.ndarray], *,
                  evaluations: int, seed: int = 42,
                  feasible: Callable[[tuple[int, ...]], bool] | None = None
                  ) -> ProgramSearchResult:
    """Sample distinct full programs and score in one batch."""
    _validate_options(options)
    if evaluations < 1:
        raise ValueError("evaluations must be positive")
    rng = np.random.default_rng(seed)
    programs = set()
    attempts = 0
    while len(programs) < evaluations:
        if attempts > evaluations * 1000:
            raise ValueError("not enough distinct compatible programs")
        attempts += 1
        choice = tuple(sorted(channel for group in options
                              for channel in group[rng.integers(len(group))]))
        if len(set(choice)) == len(choice) and (
                feasible is None or feasible(choice)):
            programs.add(choice)
    ordered = sorted(programs)
    scores = np.asarray(score(np.asarray(ordered, dtype=np.int16)),
                        dtype=np.float64)
    if scores.shape != (len(ordered),) or not np.isfinite(scores).all():
        raise ValueError("scorer returned invalid shape or values")
    best = min(range(len(ordered)), key=lambda i: (-scores[i], ordered[i]))
    return ProgramSearchResult(ordered[best], float(scores[best]),
                               evaluations, (evaluations,))
