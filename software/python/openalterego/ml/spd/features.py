"""SPD edge matrices and σ(τ) from multichannel EMG windows (Gowda paper App. B)."""

from __future__ import annotations

from typing import Iterator, List, Tuple

import numpy as np
from scipy.linalg import expm, logm

GOWDA_SPD_ETA = 0.1
GOWDA_SPD_WINDOW_MS = 100
GOWDA_SPD_STEP_MS = 50


def edge_matrix(segment_time_ch: np.ndarray) -> np.ndarray:
    """Sample covariance ℰ(τ) for window ``(time, channels)``."""
    x = np.asarray(segment_time_ch, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        c = int(x.shape[1]) if x.ndim == 2 else 1
        return np.eye(c, dtype=np.float64)
    x0 = x - x.mean(axis=0, keepdims=True)
    return (x0.T @ x0) / float(max(x.shape[0] - 1, 1))


def spd_regularize(matrix: np.ndarray, eta: float = GOWDA_SPD_ETA) -> np.ndarray:
    """ℰ ← (1−η)ℰ + η·trace(ℰ)·I (η=0.1 in paper)."""
    e = np.asarray(matrix, dtype=np.float64)
    c = int(e.shape[0])
    tr = float(np.trace(e))
    return (1.0 - float(eta)) * e + float(eta) * (tr / max(c, 1)) * np.eye(c, dtype=np.float64)


def _ensure_spd(matrix: np.ndarray, *, eta: float) -> np.ndarray:
    sym = 0.5 * (matrix + matrix.T)
    reg = spd_regularize(sym, eta=eta)
    # Tiny jitter for numerical logm stability.
    jitter = 1e-9 * float(np.trace(reg)) / max(reg.shape[0], 1)
    return reg + jitter * np.eye(reg.shape[0], dtype=np.float64)


def log_euclidean_mean(matrices: List[np.ndarray], *, eta: float = GOWDA_SPD_ETA) -> np.ndarray:
    """Fréchet mean via log-Euclidean average (Lin 2019-style SPD manifold mean)."""
    if not matrices:
        raise ValueError("need at least one matrix for Fréchet mean")
    logs = [logm(_ensure_spd(m, eta=eta)) for m in matrices]
    return np.real(expm(np.mean(logs, axis=0)))


def eigenbasis_from_frechet(frechet: np.ndarray) -> np.ndarray:
    """Return orthonormal eigenbasis Q from Fréchet mean ℱ."""
    f = np.asarray(frechet, dtype=np.float64)
    f = 0.5 * (f + f.T)
    vals, vecs = np.linalg.eigh(f)
    order = np.argsort(vals)[::-1]
    q = vecs[:, order]
    # Fix sign ambiguity for reproducibility.
    for j in range(q.shape[1]):
        if q[0, j] < 0:
            q[:, j] *= -1.0
    return q.astype(np.float64, copy=False)


def sigma_tau(edge: np.ndarray, basis_q: np.ndarray) -> np.ndarray:
    """σ(τ) = Qᵀ ℰ(τ) Q."""
    e = np.asarray(edge, dtype=np.float64)
    q = np.asarray(basis_q, dtype=np.float64)
    return q.T @ e @ q


def upper_triangle_dim(channels: int) -> int:
    c = int(channels)
    return c * (c + 1) // 2


def feature_dim_for_channels(
    channels: int,
    *,
    use_upper_tri: bool = False,
    feature_mode: str = "full",
) -> int:
    c = int(channels)
    mode = str(feature_mode or "full").strip().lower()
    if mode == "diag":
        return c
    if mode == "diag_delta":
        return c * 2
    if mode == "upper_tri" or use_upper_tri:
        return upper_triangle_dim(c)
    return c * c


def sigma_diag_vector(edge: np.ndarray, basis_q: np.ndarray, *, log_scale: bool = True) -> np.ndarray:
    """Diagonal of σ(τ) in shared eigenbasis (paper: off-diagonals ≈ 0)."""
    diag = np.diag(sigma_tau(edge, basis_q)).astype(np.float32, copy=False)
    if log_scale:
        return np.log(np.maximum(diag, 1e-8))
    return diag


def sigma_vector(
    edge: np.ndarray,
    basis_q: np.ndarray,
    *,
    use_upper_tri: bool = False,
    feature_mode: str = "full",
    log_scale: bool = True,
) -> np.ndarray:
    """Flatten σ(τ) for GRU input."""
    mode = str(feature_mode or "full").strip().lower()
    if mode in ("diag", "diag_delta"):
        return sigma_diag_vector(edge, basis_q, log_scale=log_scale)
    sig = sigma_tau(edge, basis_q).astype(np.float32, copy=False)
    if mode == "upper_tri" or use_upper_tri:
        c = int(sig.shape[0])
        return sig[np.triu_indices(c)]
    if log_scale and mode == "full":
        return np.log(np.maximum(sig.reshape(-1), 1e-8))
    return sig.reshape(-1)


def append_sigma_deltas(seq: np.ndarray) -> np.ndarray:
    """Concatenate frame-wise temporal delta: ``[σ, σ_t - σ_{t-1}]``."""
    x = np.asarray(seq, dtype=np.float32)
    if x.shape[0] < 2:
        d = np.zeros_like(x)
        return np.concatenate([x, d], axis=1).astype(np.float32, copy=False)
    d = np.zeros_like(x)
    d[1:] = x[1:] - x[:-1]
    return np.concatenate([x, d], axis=1).astype(np.float32, copy=False)


def window_samples(fs_hz: int, window_ms: int) -> int:
    return max(2, int(round(float(fs_hz) * float(window_ms) / 1000.0)))


def step_samples(fs_hz: int, step_ms: int) -> int:
    return max(1, int(round(float(fs_hz) * float(step_ms) / 1000.0)))


def iter_window_slices(
    n_samples: int,
    *,
    fs_hz: int,
    window_ms: int = GOWDA_SPD_WINDOW_MS,
    step_ms: int = GOWDA_SPD_STEP_MS,
) -> Iterator[Tuple[int, int]]:
    win = window_samples(int(fs_hz), int(window_ms))
    step = step_samples(int(fs_hz), int(step_ms))
    if n_samples < win:
        return
    for start in range(0, n_samples - win + 1, step):
        yield start, start + win


def sliding_edge_matrices(
    segment_time_ch: np.ndarray,
    *,
    fs_hz: int,
    window_ms: int = GOWDA_SPD_WINDOW_MS,
    step_ms: int = GOWDA_SPD_STEP_MS,
    eta: float = GOWDA_SPD_ETA,
) -> List[np.ndarray]:
    """Edge matrices for each sliding window in a segment."""
    seg = np.asarray(segment_time_ch, dtype=np.float64)
    out: List[np.ndarray] = []
    for s0, s1 in iter_window_slices(int(seg.shape[0]), fs_hz=int(fs_hz), window_ms=window_ms, step_ms=step_ms):
        e = edge_matrix(seg[s0:s1, :])
        out.append(spd_regularize(e, eta=float(eta)))
    return out


def segment_to_sigma_sequence(
    segment_time_ch: np.ndarray,
    basis_q: np.ndarray,
    *,
    fs_hz: int,
    window_ms: int = GOWDA_SPD_WINDOW_MS,
    step_ms: int = GOWDA_SPD_STEP_MS,
    eta: float = GOWDA_SPD_ETA,
    use_upper_tri: bool = False,
    use_delta: bool = False,
    feature_mode: str = "full",
) -> np.ndarray:
    """σ(τ) sequence ``(T_frames, D)`` for one word/event segment."""
    mode = str(feature_mode or "full").strip().lower()
    if use_upper_tri and mode == "full":
        mode = "upper_tri"
    edges = sliding_edge_matrices(
        segment_time_ch,
        fs_hz=int(fs_hz),
        window_ms=int(window_ms),
        step_ms=int(step_ms),
        eta=float(eta),
    )
    c = int(basis_q.shape[0])
    d = feature_dim_for_channels(c, feature_mode=mode)
    if not edges:
        return np.zeros((1, d * (2 if mode == "diag_delta" or use_delta else 1)), dtype=np.float32)
    rows = [sigma_vector(e, basis_q, feature_mode=mode) for e in edges]
    seq = np.stack(rows, axis=0).astype(np.float32, copy=False)
    if mode == "diag_delta" or use_delta:
        seq = append_sigma_deltas(seq)
    return seq
