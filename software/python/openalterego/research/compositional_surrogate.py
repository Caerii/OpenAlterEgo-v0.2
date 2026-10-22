"""Sparse additive surrogate for scores of complete grammar programs."""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr


@dataclass(frozen=True)
class AdditiveProgramModel:
    intercept: float
    offsets: np.ndarray
    coefficients: np.ndarray

    def predict(self, codes: np.ndarray) -> np.ndarray:
        codes = np.asarray(codes)
        sizes = np.diff(self.offsets)
        if (codes.ndim != 2 or codes.shape[1] != len(sizes) or
                not np.issubdtype(codes.dtype, np.integer) or
                np.any(codes < 0) or np.any(codes >= sizes)):
            raise ValueError("invalid grammar codes")
        return self.intercept + self.coefficients[
            codes + self.offsets[:-1]].sum(axis=1)


def fit_additive_program_model(codes: np.ndarray, scores: np.ndarray,
                               option_sizes: tuple[int, ...], *,
                               ridge: float = 10.0) -> AdditiveProgramModel:
    """Fit regularized part effects from exact complete-program scores.

    The model is a compute surrogate for the *selection score*, not a
    prediction of future-session accuracy or a causal physiology model.
    """
    codes = np.asarray(codes)
    scores = np.asarray(scores, dtype=np.float64)
    sizes = np.asarray(option_sizes, dtype=np.int64)
    if (codes.ndim != 2 or codes.shape[0] < 2 or
            codes.shape[1] != len(sizes) or not len(sizes) or
            scores.shape != (len(codes),) or
            not np.issubdtype(codes.dtype, np.integer) or
            np.any(sizes < 1) or np.any(codes < 0) or
            np.any(codes >= sizes) or not np.isfinite(scores).all() or
            not np.isfinite(ridge) or ridge <= 0):
        raise ValueError("invalid additive program fit")
    offsets = np.r_[0, np.cumsum(sizes)]
    n, parts = codes.shape
    matrix = csr_matrix((np.ones(n * parts),
                         (np.repeat(np.arange(n), parts),
                          (codes + offsets[:-1]).ravel())),
                        shape=(n, int(offsets[-1])))
    intercept = float(scores.mean())
    coefficients = lsqr(matrix, scores - intercept,
                        damp=np.sqrt(ridge), atol=1e-8, btol=1e-8,
                        iter_lim=500)[0]
    return AdditiveProgramModel(intercept, offsets, coefficients)
