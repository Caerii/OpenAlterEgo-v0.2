"""Cross-fitted donor calibration for ranked measurement-program decisions."""

import numpy as np


def predict_left_out_person(proxies: np.ndarray, outcomes: np.ndarray,
                            target: int, *, ridge: float = 100.0
                            ) -> tuple[np.ndarray, dict]:
    """Predict candidate query deviations without the target's query labels.

    proxies has axes person, candidate, (day-one CV, mean support transfer).
    Every donor training row receives a prior from *other* donors, excluding
    both itself and the outer target.
    """
    proxies = np.asarray(proxies, dtype=np.float64)
    outcomes = np.asarray(outcomes, dtype=np.float64)
    if (proxies.ndim != 3 or proxies.shape[2] != 2 or
            outcomes.shape != proxies.shape[:2] or
            len(proxies) < 3 or not 0 <= target < len(proxies) or
            proxies.shape[1] < 2 or not np.isfinite(proxies).all() or
            not np.isfinite(ridge) or
            ridge <= 0):
        raise ValueError("invalid cross-person calibration inputs")
    donors = np.asarray([i for i in range(len(proxies)) if i != target])
    donor_outcomes = outcomes[donors]
    if not np.isfinite(donor_outcomes).all():
        raise ValueError("nonfinite donor outcomes")
    donor_sum = donor_outcomes.sum(axis=0)
    donor_features = []
    donor_targets = []
    for local, person in enumerate(donors):
        prior = (donor_sum - outcomes[person]) / (len(donors) - 1)
        features = np.column_stack((proxies[person], prior))
        donor_features.append(features - features.mean(axis=0))
        donor_targets.append(outcomes[person] - outcomes[person].mean())
    x = np.vstack(donor_features)
    y = np.concatenate(donor_targets)
    scale = np.maximum(x.std(axis=0), 1e-8)
    x /= scale
    coefficients = np.linalg.solve(x.T @ x + ridge * np.eye(3), x.T @ y)
    target_prior = donor_outcomes.mean(axis=0)
    target_features = np.column_stack((proxies[target], target_prior))
    target_features = (target_features - target_features.mean(axis=0)) / scale
    prediction = target_features @ coefficients
    return prediction, {
        "donor_indices": donors.tolist(),
        "coefficients_standardized": coefficients.tolist(),
        "feature_scale": scale.tolist(),
        "donor_prior": target_prior,
    }


def nested_policy_gate(proxies: np.ndarray, outcomes: np.ndarray,
                       target: int, *, seed: int = 77,
                       draws: int = 2000) -> dict:
    """Permit a donor policy only after positive inner person-level evidence.

    The outer target's query outcomes are never read. Inner pseudo-targets
    each train on the other available donors, with the same program pool.
    """
    proxies = np.asarray(proxies, dtype=np.float64)
    outcomes = np.asarray(outcomes, dtype=np.float64)
    if (proxies.ndim != 3 or outcomes.shape != proxies.shape[:2] or
            len(proxies) < 4 or not 0 <= target < len(proxies) or
            draws < 1):
        raise ValueError("invalid nested gate inputs")
    donors = np.asarray([i for i in range(len(proxies)) if i != target])
    inner_proxies, inner_outcomes = proxies[donors], outcomes[donors]
    if not np.isfinite(inner_outcomes).all():
        raise ValueError("nonfinite inner donor outcomes")
    gains = {"donor_only": [], "donor_calibrated": []}
    for inner_target in range(len(donors)):
        learned, model = predict_left_out_person(
            inner_proxies, inner_outcomes, inner_target)
        baseline = .5 * inner_proxies[inner_target].sum(axis=1)
        baseline_score = inner_outcomes[inner_target, int(np.argmax(baseline))]
        for policy, scores in (("donor_only", model["donor_prior"]),
                               ("donor_calibrated", learned)):
            gains[policy].append(float(
                inner_outcomes[inner_target, int(np.argmax(scores))] -
                baseline_score))
    rng = np.random.default_rng(seed)
    bootstrap_indices = rng.integers(len(donors), size=(draws, len(donors)))
    evidence = {}
    eligible = []
    for policy, values in gains.items():
        values = np.asarray(values)
        lower = float(np.quantile(values[bootstrap_indices].mean(axis=1), .05))
        mean = float(values.mean())
        evidence[policy] = {"inner_mean_gain": mean,
                            "inner_wins": int(np.count_nonzero(values > 0)),
                            "inner_losses": int(np.count_nonzero(values < 0)),
                            "bootstrap_5th_percentile_gain": lower}
        if lower > 0:
            eligible.append(policy)
    selected = (max(eligible, key=lambda policy: evidence[policy]["inner_mean_gain"])
                if eligible else "two_support")
    return {"selected_policy": selected,
            "outer_target_excluded_from_inner": int(target),
            "inner_donor_indices": donors.tolist(),
            "bootstrap_seed": seed, "bootstrap_draws": draws,
            "evidence": evidence}
