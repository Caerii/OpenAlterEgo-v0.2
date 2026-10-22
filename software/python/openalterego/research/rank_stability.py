"""Descriptive transport diagnostics for ranked recorded-channel programs."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def compare_rankings(selection_scores: dict, later_scores: dict,
                     *, top_sizes: tuple[int, ...] = (16, 100)) -> dict:
    """Compare fixed-program scores without treating overlapping subsets as IID."""
    if selection_scores.keys() != later_scores.keys() or len(selection_scores) < 2:
        raise ValueError("rankings must score the same programs")
    keys = sorted(selection_scores)
    first = np.asarray([selection_scores[key] for key in keys], dtype=float)
    later = np.asarray([later_scores[key] for key in keys], dtype=float)
    if not np.isfinite(first).all() or not np.isfinite(later).all():
        raise ValueError("ranking scores must be finite")
    first_ranks = rankdata(first, method="average")
    later_ranks = rankdata(later, method="average")
    correlation = np.corrcoef(first_ranks, later_ranks)[0, 1]
    ordered_first = sorted(keys, key=lambda key: (-selection_scores[key], key))
    ordered_later = sorted(keys, key=lambda key: (-later_scores[key], key))
    winner = ordered_first[0]
    return {
        "programs": len(keys),
        "spearman_score_rank": float(correlation) if np.isfinite(correlation) else None,
        "top_overlap": {str(size): len(set(ordered_first[:size]) &
                                       set(ordered_later[:size]))
                        for size in top_sizes},
        "selection_winner": list(winner),
        "selection_winner_selection_score": float(selection_scores[winner]),
        "selection_winner_later_score": float(later_scores[winner]),
        "later_proxy_oracle_score": float(later_scores[ordered_later[0]]),
        "later_proxy_regret": float(later_scores[ordered_later[0]] - later_scores[winner]),
    }


def batch_robust_choices(batch_scores: list[dict], later_scores: dict,
                         *, penalties: tuple[float, ...] = (0.0, 0.5, 1.0)) -> dict:
    """Retrospectively check whether within-session instability helps selection."""
    if len(batch_scores) < 2 or any(row.keys() != later_scores.keys()
                                     for row in batch_scores):
        raise ValueError("batches must score the same programs")
    keys = sorted(later_scores)
    scores = np.asarray([[batch[key] for key in keys]
                         for batch in batch_scores], dtype=float)
    if (not np.isfinite(scores).all() or
            not np.isfinite(penalties).all() or
            any(penalty < 0 for penalty in penalties)):
        raise ValueError("scores and penalties must be finite and nonnegative")
    mean, std = scores.mean(axis=0), scores.std(axis=0)
    oracle = max(later_scores.values())
    result = {}
    for penalty in penalties:
        utility = mean - penalty * std
        winner_index = min(range(len(keys)), key=lambda i: (-utility[i], keys[i]))
        winner = keys[winner_index]
        result[str(penalty)] = {
            "choice": list(winner),
            "batch_mean": float(mean[winner_index]),
            "batch_std": float(std[winner_index]),
            "later_proxy_score": float(later_scores[winner]),
            "later_proxy_regret": float(oracle - later_scores[winner]),
        }
    return result
