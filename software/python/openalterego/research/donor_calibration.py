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
