"""Map literature montage site names to 1D pickup geometry for forward models."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .biophysical.forward_model import green_pickup_matrix

# Normalized arc position along face/neck (0 = superior face, 1 = inferior neck).
_SITE_X: Dict[str, float] = {
    "laryngeal": 0.48,
    "hyoid": 0.55,
    "levator_anguli_oris": 0.22,
    "orbicularis_oris": 0.30,
    "platysma": 0.72,
    "digastric": 0.58,
    "mentum": 0.38,
    "masseter_or_scm": 0.82,
    "face_1": 0.18,
    "face_2": 0.26,
    "face_3": 0.34,
    "face_4": 0.42,
    "neck_1": 0.62,
    "neck_2": 0.70,
    "neck_3": 0.78,
    "neck_4": 0.86,
    "LAO": 0.24,
    "DAO": 0.32,
    "BUC": 0.40,
    "ABD": 0.50,
    "ZM": 0.28,
    "ZYG": 0.20,
    "RIS": 0.28,
    "SCM": 0.76,
    "PLT": 0.68,
    "site_7": 0.44,
    "site_8": 0.52,
    "headphone_emg_1": 0.35,
    "headphone_emg_2": 0.45,
    "headphone_emg_3": 0.55,
    "headphone_emg_4": 0.65,
}
for _i in range(1, 32):
    _SITE_X[f"gowda_site_{_i}"] = 0.12 + 0.76 * float(_i - 1) / 30.0


def _positions_for_sites(sites: List[str]) -> np.ndarray:
    xs: List[float] = []
    for i, site in enumerate(sites):
        if site in _SITE_X:
            xs.append(float(_SITE_X[site]))
        elif site.startswith("neck_diff_"):
            # SilentWear band: spread inferiorly along neck.
            try:
                idx = int(site.split("_")[-1])
            except ValueError:
                idx = i + 1
            xs.append(0.58 + 0.035 * float(idx))
        else:
            xs.append(float(i) / max(1, len(sites) - 1))
    arr = np.asarray(xs, dtype=np.float64)
    if arr.size == 1:
        return arr.astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-6:
        return np.linspace(0.2, 0.8, arr.size, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def electrode_positions_1d(montage_name: str) -> np.ndarray:
    """Return electrode positions in [0, 1] for a named montage preset."""
    from ..hardware.montages import get_montage

    m = get_montage(str(montage_name))
    return _positions_for_sites(list(m.sites))


def green_pickup_matrix_from_montage(
    montage_name: str,
    n_sources: int = 14,
    *,
    falloff: float = 0.16,
) -> np.ndarray:
    """Build G (n_electrodes, n_sources) using montage site geometry."""
    xe = electrode_positions_1d(montage_name)
    ne = int(xe.size)
    ns = max(1, int(n_sources))
    xs = np.linspace(float(xe.min()), float(xe.max()), ns, dtype=np.float64)
    ell = max(float(falloff), 0.03)
    d = np.abs(xe[:, None].astype(np.float64) - xs[None, :])
    g = np.exp(-d / ell)
    g /= np.sum(g, axis=0, keepdims=True) + 1e-12
    return g.astype(np.float32)


def resolve_forward_pickup_matrix(
    n_electrodes: int,
    n_sources: int,
    *,
    montage_name: Optional[str] = None,
    falloff: float = 0.16,
) -> np.ndarray:
    """Montage-aware G when possible; else uniform 1D layout."""
    if montage_name:
        try:
            g = green_pickup_matrix_from_montage(montage_name, n_sources, falloff=falloff)
            if g.shape[0] == int(n_electrodes):
                return g
        except ValueError:
            pass
    return green_pickup_matrix(int(n_electrodes), int(n_sources), falloff=falloff)
