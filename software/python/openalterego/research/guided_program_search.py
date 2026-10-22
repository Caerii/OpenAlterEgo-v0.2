"""Budget-matched exploration of complete compositional measurement programs."""

from collections.abc import Callable, Sequence

import numpy as np


def guided_search(options: Sequence[Sequence[tuple[int, ...]]],
                  score: Callable[[np.ndarray], np.ndarray], *,
                  budget: int = 5000, seed: int = 42,
                  batch_size: int = 500, elite_count: int = 32
                  ) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    """Seed randomly, then mix local edits, recombination, and exploration.

    Returns every evaluated complete program and score, plus the best index at
    each batch boundary. No partial-program score is assumed to mean anything.
    """
    if (not options or any(not group for group in options) or budget < 1 or
            batch_size < 1 or elite_count < 1):
        raise ValueError("invalid grammar or search budget")
    rng = np.random.default_rng(seed)
    parts = len(options)
    channel_to_group = {}
    option_lookup = []
    for group_index, group in enumerate(options):
        lookup = {}
        for option_index, option in enumerate(group):
            key = tuple(sorted(option))
            if key in lookup or len(key) != len(set(key)):
                raise ValueError("duplicate or invalid grammar option")
            lookup[key] = option_index
            for channel in key:
                channel_to_group.setdefault(channel, set()).add(group_index)
        option_lookup.append(lookup)

    def encode(indices):
        return tuple(sorted(channel for group, index in zip(options, indices)
                            for channel in group[index]))

    def decode(program):
        groups = [[] for _ in options]
        for channel in program:
            owners = channel_to_group.get(channel, ())
            if len(owners) != 1:
                raise ValueError("channels must identify a unique program part")
            groups[next(iter(owners))].append(channel)
        return tuple(option_lookup[i][tuple(group)]
                     for i, group in enumerate(groups))

    seen = set()
    programs = []
    scores = []
    winners = {}
    while len(programs) < budget:
        target = min(batch_size, budget - len(programs))
        proposed = []
        attempts = 0
        elites = (sorted(range(len(programs)),
                         key=lambda i: (-scores[i], programs[i]))[:elite_count]
                  if programs else [])
        while len(proposed) < target:
            attempts += 1
            if attempts > target * 1000:
                raise ValueError("unable to sample enough distinct programs")
            if not elites or rng.random() < .25:
                indices = [int(rng.integers(len(group))) for group in options]
            elif rng.random() < .65 or len(elites) == 1:
                parent = decode(programs[int(rng.choice(elites))])
                indices = list(parent)
                part = int(rng.integers(parts))
                indices[part] = int(rng.integers(len(options[part])))
            else:
                first = decode(programs[int(rng.choice(elites))])
                second = decode(programs[int(rng.choice(elites))])
                indices = [a if rng.random() < .5 else b
                           for a, b in zip(first, second)]
            candidate = encode(indices)
            if len(set(candidate)) != len(candidate) or candidate in seen:
                continue
            seen.add(candidate)
            proposed.append(candidate)
        array = np.asarray(proposed, dtype=np.int16)
        values = np.asarray(score(array), dtype=np.float64)
        if values.shape != (target,) or not np.isfinite(values).all():
            raise ValueError("scorer returned invalid scores")
        programs.extend(proposed)
        scores.extend(map(float, values))
        winners[len(programs)] = min(range(len(programs)),
                                     key=lambda i: (-scores[i], programs[i]))
    return np.asarray(programs, dtype=np.int16), np.asarray(scores), winners
