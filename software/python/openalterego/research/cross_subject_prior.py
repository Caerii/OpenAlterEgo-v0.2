"""Leave-one-subject-out priors over already recorded channel subsets."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def _standardize(values: np.ndarray) -> np.ndarray:
    scale = values.std()
    return (values - values.mean()) / scale if scale > 0 else np.zeros_like(values)


def additive_channel_prior(programs: list[tuple[str, tuple[int, ...]]],
                           donor_scores: np.ndarray, *, alpha: float = 100.0
                           ) -> np.ndarray:
    """Shrink a donor score surface to normalization and channel main effects."""
    if len(programs) != len(donor_scores) or alpha <= 0:
        raise ValueError("programs and scores must align")
    design = np.zeros((len(programs), 16), dtype=float)
    design[:, 0] = 1
    for index, (normalization, channels) in enumerate(programs):
        if normalization not in ("none", "recording_robust") or len(channels) != 6:
            raise ValueError("unexpected montage program")
        design[index, 1] = normalization == "recording_robust"
        design[index, np.asarray(channels) + 2] = 1
    ridge = np.eye(16) * alpha
    ridge[0, 0] = 0
    coefficients = np.linalg.solve(design.T @ design + ridge,
                                   design.T @ donor_scores)
    return design @ coefficients


def leave_one_out_prior(programs: list[tuple[str, tuple[int, ...]]],
                        personal: dict[str, np.ndarray],
                        future: dict[str, np.ndarray]) -> dict:
    """For each target, use only other subjects' future scores as a prior."""
    subjects = sorted(personal)
    if (len(subjects) < 3 or subjects != sorted(future) or
            any(len(personal[s]) != len(programs) or
                len(future[s]) != len(programs) for s in subjects)):
        raise ValueError("need aligned scores from at least three subjects")
    if any(not np.isfinite(personal[s]).all() or
           not np.isfinite(future[s]).all() for s in subjects):
        raise ValueError("scores must be finite")
    results = {}
    for target in subjects:
        donors = [subject for subject in subjects if subject != target]
        donor_scores = np.mean([future[subject] for subject in donors], axis=0)
        additive = additive_channel_prior(programs, donor_scores)
        candidates = {
            "personal": personal[target],
            "donor_raw": donor_scores,
            "donor_additive": additive,
            "blend_half": (_standardize(personal[target]) +
                           _standardize(additive)) / 2,
        }
        future_scores = future[target]
        oracle = float(future_scores.max())
        methods = {}
        for name, values in candidates.items():
            winner = int(np.argmax(values))
            methods[name] = {
                "program": [programs[winner][0], list(programs[winner][1])],
                "target_future_proxy_score": float(future_scores[winner]),
                "target_future_proxy_regret": float(oracle - future_scores[winner]),
            }
        spread = programs.index(("recording_robust", (0, 2, 5, 8, 11, 13)))
        methods["fixed_spread_robust"] = {
            "program": [programs[spread][0], list(programs[spread][1])],
            "target_future_proxy_score": float(future_scores[spread]),
            "target_future_proxy_regret": float(oracle - future_scores[spread]),
        }
        a, b = rankdata(donor_scores), rankdata(future_scores)
        correlation = (float(np.corrcoef(a, b)[0, 1])
                       if a.std() > 0 and b.std() > 0 else None)
        results[target] = {
            "donors": donors,
            "donor_target_spearman": correlation,
            "methods": methods,
        }
    return results
