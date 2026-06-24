"""SPD geometry features for Gowda EMG (edge matrices, Fréchet mean, σ(τ))."""

from .basis import SPDBasis, fit_spd_basis_from_events, load_spd_basis, save_spd_basis
from .features import edge_matrix, log_euclidean_mean, spd_regularize, sigma_tau, sliding_edge_matrices

__all__ = [
    "SPDBasis",
    "edge_matrix",
    "fit_spd_basis_from_events",
    "load_spd_basis",
    "log_euclidean_mean",
    "save_spd_basis",
    "sigma_tau",
    "sliding_edge_matrices",
    "spd_regularize",
]
