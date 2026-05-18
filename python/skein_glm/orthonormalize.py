"""Per-block (group) orthonormalization helpers (grpreg `orthogonalize`).

For a design matrix `X` partitioned by groups, this module produces an
orthonormalized matrix `X_orth` where every block satisfies
``X_orth_g.T @ X_orth_g / n == I``. Solvers operating on `X_orth` see a
clean per-block Lipschitz of 1 and a closed-form block soft-threshold
prox — matching grpreg's `gdfit_*` C kernels.

The transformation is invertible: fit on `X_orth`, then call
:meth:`BlockBackTransform.apply_to_coefs_path` to convert coefficients
back to original-feature scale. The high-level
:func:`fit_with_orthonormalization` wrapper handles the
centering + back-transform + intercept reconstruction in one shot.

References
----------
Breheny, P., & Huang, J. (2009). Penalized methods for bi-level
variable selection. *Statistics and Its Interface* 2(3), 369–380.
"""
from __future__ import annotations

from typing import Any, Protocol, TypeVar

import numpy as np
from numpy.typing import NDArray

from skein_glm import _core


class BlockBackTransform:
    """Per-block back-transform `T_g` for a group-orthonormalized design.

    Constructed by :func:`orthonormalize_groups` alongside the
    orthonormalized matrix. Wraps the packed list-of-tuples form
    returned by `_core.orthonormalize_groups_dense` and re-emits the
    transform via the Rust binding (vectorized over a path).
    """

    def __init__(self, packed):
        self._packed = packed

    @property
    def n_groups(self) -> int:
        """Number of groups in the partition."""
        return len(self._packed)

    def apply_to_coefs(self, beta_orth: NDArray[np.float64]) -> NDArray[np.float64]:
        """Map a single β vector from orthonormalized to original space."""
        beta_orth = np.ascontiguousarray(beta_orth, dtype=np.float64)
        return _core.back_transform_coefs(beta_orth, self._packed)

    def apply_to_coefs_path(
        self, betas_orth: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Map an `(n_lambdas, n_features)` coefficient path back to
        original-feature space."""
        betas_orth = np.ascontiguousarray(betas_orth, dtype=np.float64)
        return _core.back_transform_coefs_path(betas_orth, self._packed)


def orthonormalize_groups(
    x: NDArray[np.float64], groups: NDArray[np.int64]
) -> tuple[NDArray[np.float64], BlockBackTransform]:
    """Orthonormalize a dense design matrix block-by-block.

    Parameters
    ----------
    x : ndarray of shape ``(n_samples, n_features)``
        Dense design matrix. Sparse / mmap / chunked backends are not
        supported in this version — the SVD-equivalent per-block
        Cholesky requires materialized columns.
    groups : ndarray of int64 of shape ``(n_features,)``
        Per-feature group labels. Labels must form a contiguous range
        ``0..n_groups``.

    Returns
    -------
    x_orth : ndarray of shape ``(n_samples, n_features)``
        Block-orthonormalized design: ``x_orth_g.T @ x_orth_g / n == I``
        for every group ``g``.
    back_transform : :class:`BlockBackTransform`
        Per-group `T_g` matrices. Use ``back_transform.apply_to_coefs_path``
        to map fitted coefficients back to original-feature scale.

    Raises
    ------
    ValueError
        If any group's Gram matrix is rank-deficient (perfectly collinear
        columns within the group). Drop the dependent column first.
    """
    x = np.ascontiguousarray(x, dtype=np.float64)
    groups = np.ascontiguousarray(groups, dtype=np.int64)
    if x.ndim != 2:
        raise ValueError(f"x must be 2D, got shape {x.shape}")
    if groups.shape[0] != x.shape[1]:
        raise ValueError(
            f"groups length {groups.shape[0]} does not match n_features {x.shape[1]}"
        )
    x_orth, packed = _core.orthonormalize_groups_dense(x, groups)
    return x_orth, BlockBackTransform(packed)


# Structural type for the group-path estimators that
# `fit_with_orthonormalization` accepts. We can't import the estimator
# classes here without a circular import; the Protocol captures the
# minimum-needed interface.
class _GroupPathEstimator(Protocol):
    coefs_: NDArray[np.float64]
    intercepts_: NDArray[np.float64]
    lambdas_: NDArray[np.float64]
    info_: dict[str, Any]
    fit_intercept: bool
    standardize: bool

    def fit(self, x, y): ...


EstT = TypeVar("EstT", bound=_GroupPathEstimator)


def fit_with_orthonormalization(
    estimator: EstT,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    groups: NDArray[np.int64],
    *,
    fit_intercept: bool = True,
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], BlockBackTransform
]:
    """Fit a group-path estimator on block-orthonormalized X, then
    back-transform coefficients into original-feature space.

    This is grpreg's `grpreg()` pipeline distilled: center → orthonormalize
    → fit on `X_orth` with no internal centering/scaling → back-transform
    → reconstruct intercept from column means. Works with any
    *PathRegressor that exposes ``coefs_``, ``intercepts_``, ``lambdas_``
    and accepts ``fit_intercept`` / ``standardize`` constructor flags.

    The estimator is fit with ``fit_intercept=False`` and
    ``standardize=False`` regardless of how it was originally configured
    — those preprocessing steps are subsumed by the centering +
    orthonormalization we apply here. We mutate those two attributes on
    the estimator instance and restore them after fit.

    Parameters
    ----------
    estimator : group-path estimator instance
        Will be fit in-place; its ``coefs_`` / ``intercepts_`` /
        ``lambdas_`` / ``info_`` reflect the **orthonormalized** fit
        after this call (not the original-space versions returned).
    x : ndarray of shape ``(n_samples, n_features)``
    y : ndarray of shape ``(n_samples,)``
    groups : ndarray of int64 of shape ``(n_features,)``
    fit_intercept : bool, default True
        Whether to center X and y before orthonormalization, and
        reconstruct an intercept ``ȳ − x̄ @ coefs_orig.T`` per λ.

    Returns
    -------
    coefs_orig : ndarray of shape ``(n_lambdas, n_features)``
        Coefficients in original-feature scale.
    intercepts : ndarray of shape ``(n_lambdas,)``
        Per-λ intercepts (zeros when ``fit_intercept=False``).
    lambdas : ndarray of shape ``(n_lambdas,)``
        The λ values used by the underlying solver.
    back_transform : :class:`BlockBackTransform`
        The per-group transform used; kept for downstream predictions.
    """
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    groups = np.ascontiguousarray(groups, dtype=np.int64)
    if x.ndim != 2:
        raise ValueError(f"x must be 2D, got shape {x.shape}")
    if y.ndim != 1 or y.shape[0] != x.shape[0]:
        raise ValueError(f"y must be 1D with length {x.shape[0]}, got {y.shape}")

    if fit_intercept:
        x_mean = x.mean(axis=0)
        y_mean = float(y.mean())
        x_c = x - x_mean
        y_c = y - y_mean
    else:
        x_mean = None
        y_mean = 0.0
        x_c = x
        y_c = y

    x_orth, bt = orthonormalize_groups(x_c, groups)

    # Temporarily disable the estimator's internal centering / scaling —
    # we've subsumed both via centering + orthonormalization.
    saved_fit_intercept = estimator.fit_intercept
    saved_standardize = estimator.standardize
    estimator.fit_intercept = False
    estimator.standardize = False
    try:
        estimator.fit(x_orth, y_c)
    finally:
        estimator.fit_intercept = saved_fit_intercept
        estimator.standardize = saved_standardize

    coefs_orig = bt.apply_to_coefs_path(estimator.coefs_)
    if fit_intercept:
        # intercept_k = ȳ − x̄ · coefs_orig[k]
        intercepts = y_mean - coefs_orig @ x_mean
    else:
        intercepts = np.zeros(coefs_orig.shape[0])
    return coefs_orig, intercepts, estimator.lambdas_, bt
