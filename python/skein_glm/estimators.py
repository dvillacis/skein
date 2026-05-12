"""sklearn-compatible estimators built on the Rust core.

`MCPRegressor` / `SCADRegressor` are single-λ wrappers; `MCPPathRegressor` /
`SCADPathRegressor` return the entire λ-path. All four standardize and fit
an intercept by default (matching `glmnet` / `ncvreg` conventions) and
expose `coef_`, `intercept_`, and `info_` after `fit`.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, RegressorMixin

from skein_glm import _core
from skein_glm.mmap import (
    ChunkedDesignF32,
    ChunkedDesignF64,
    MmapDesignF32,
    MmapDesignF64,
)


def _is_mmap(x) -> bool:
    return isinstance(x, (MmapDesignF64, MmapDesignF32))


def _is_chunked(x) -> bool:
    return isinstance(x, (ChunkedDesignF64, ChunkedDesignF32))


def _is_sparse(x) -> bool:
    """True if `x` is a scipy.sparse matrix-like object. scipy is imported
    lazily so dense-only users don't pay the import cost."""
    try:
        from scipy import sparse  # type: ignore[import-untyped]
    except ImportError:
        return False
    return sparse.issparse(x)


def _validate_y_binary(y_arr):
    if not np.all((y_arr == 0.0) | (y_arr == 1.0)):
        raise ValueError("logistic regression requires y ∈ {0, 1}")


def _validate_y_nonneg(y_arr):
    if not np.all(np.isfinite(y_arr)) or np.any(y_arr < 0.0):
        raise ValueError("Poisson regression requires y ≥ 0 (finite)")


def _validate_sample_weights(sw, x):
    """Coerce optional per-sample weights into a contiguous float64
    array; validate length, finiteness, and non-negativity.

    Returns `None` when no weights were supplied. Length is read from
    whichever attribute the design exposes (`n_rows` for chunked/mmap,
    `shape[0]` otherwise)."""
    if sw is None:
        return None
    sw_arr = np.ascontiguousarray(sw, dtype=np.float64)
    n = x.n_rows if (_is_chunked(x) or _is_mmap(x)) else x.shape[0]
    if sw_arr.ndim != 1 or sw_arr.shape[0] != n:
        raise ValueError(
            f"sample_weights must be 1D with length {n}, got shape {sw_arr.shape}"
        )
    if not np.all(np.isfinite(sw_arr)) or np.any(sw_arr < 0.0):
        raise ValueError("sample_weights must be finite and non-negative")
    return sw_arr


def _reject_sample_weights_on_non_dense(sw, x, *, estimator_kind: str):
    """Per-sample weights are wired only on the dense PyO3 entries in
    this round. Raise a helpful error if the user combines them with
    sparse / chunked / mmap X — the latter would silently drop the
    kwarg and return the unweighted answer, which is the original
    "couldn't see the effect" symptom."""
    if sw is None:
        return
    if _is_sparse(x) or _is_chunked(x) or _is_mmap(x):
        raise NotImplementedError(
            f"{estimator_kind}: sample_weights is currently supported only for "
            "dense X (sparse / chunked / mmap pending). Convert with `x.toarray()` "
            "or leave sample_weights unset."
        )


def _as_csc_arrays(x):
    """Return `(data, indices, indptr, n_rows, n_cols)` from a scipy
    sparse matrix in CSC layout. Converts other sparse formats to CSC."""
    from scipy import sparse  # type: ignore[import-untyped]
    if not sparse.isspmatrix_csc(x):
        x = x.tocsc()
    data = np.ascontiguousarray(x.data, dtype=np.float64)
    indices = np.ascontiguousarray(x.indices, dtype=np.int64)
    indptr = np.ascontiguousarray(x.indptr, dtype=np.int64)
    n_rows, n_cols = x.shape
    return data, indices, indptr, int(n_rows), int(n_cols)


class _NonconvexRegressorBase(BaseEstimator, RegressorMixin):
    info_: dict[str, Any]
    coef_: NDArray[np.float64]
    intercept_: float
    n_features_in_: int

    def _validate_xy(
        self, x: NDArray[np.float64], y: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        x = np.ascontiguousarray(x, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"x must be 2D, got shape {x.shape}")
        if y.ndim != 1 or y.shape[0] != x.shape[0]:
            raise ValueError(
                f"y must be 1D with length {x.shape[0]}, got shape {y.shape}"
            )
        return x, y

    def predict(self, x) -> NDArray[np.float64]:
        if _is_sparse(x):
            return np.asarray(x @ self.coef_).ravel() + self.intercept_
        x = np.ascontiguousarray(x, dtype=np.float64)
        return x @ self.coef_ + self.intercept_


class MCPRegressor(_NonconvexRegressorBase):
    """MCP-penalized least squares at a single λ.

    The single-λ companion to :class:`MCPPathRegressor`. Use this
    when you've already chosen ``lambda_`` (via external CV, prior
    knowledge, etc.); use :class:`MCPPathRegressor` or
    :class:`skein_glm.MCPPathCV` when you want to fit a full path or
    have skein_glm pick λ for you.

    Parameters
    ----------
    lambda_ : float, default 0.1
        Regularization strength. Larger → more sparsity.
    gamma : float, default 3.0
        MCP nonconvexity (>1). ``gamma=1e6`` is ≈ lasso.
    weights : array-like of shape (n_features,) or None
        Per-feature penalty weights.
    max_iter : int, default 100
    tol : float, default 1e-6
    fit_intercept : bool, default True
    standardize : bool, default False
    screening : {"off", "strong", "gap_safe"}, default "strong"
    acceleration : int or None, default 5

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
    intercept_ : float
    info_ : dict
    n_features_in_ : int

    See Also
    --------
    skein_glm.MCPPathRegressor : Full λ-path with warm starts.
    """

    def __init__(
        self,
        lambda_: float = 0.1,
        gamma: float = 3.0,
        *,
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        screening: str = "strong",
        acceleration: int | None = 5,
    ) -> None:
        self.lambda_ = lambda_
        self.gamma = gamma
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.screening = screening
        self.acceleration = acceleration

    def fit(self, x, y: NDArray[np.float64]) -> "MCPRegressor":
        w = (
            np.ascontiguousarray(self.weights, dtype=np.float64)
            if self.weights is not None
            else None
        )
        if _is_sparse(x):
            y = np.ascontiguousarray(y, dtype=np.float64)
            data, indices, indptr, n_rows, n_cols = _as_csc_arrays(x)
            if y.ndim != 1 or y.shape[0] != n_rows:
                raise ValueError(
                    f"y must be 1D with length {n_rows}, got shape {y.shape}"
                )
            coefs, intercepts, _, info = _core.solve_mcp_ls_path_sparse(
                data, indices, indptr, n_rows, n_cols, y,
                gamma=self.gamma,
                lambdas=np.array([self.lambda_], dtype=np.float64),
                weights=w,
                max_iter=self.max_iter,
                tol=self.tol,
                screening=self.screening,
                acceleration=self.acceleration,
                fit_intercept=self.fit_intercept,
                standardize_x=self.standardize,
            )
            self.n_features_in_ = n_cols
        else:
            x, y = self._validate_xy(x, y)
            coefs, intercepts, _, info = _core.solve_mcp_ls_path(
                x,
                y,
                gamma=self.gamma,
                lambdas=np.array([self.lambda_], dtype=np.float64),
                weights=w,
                max_iter=self.max_iter,
                tol=self.tol,
                screening=self.screening,
                acceleration=self.acceleration,
                fit_intercept=self.fit_intercept,
                standardize_x=self.standardize,
            )
            self.n_features_in_ = x.shape[1]
        self.coef_ = coefs[0]
        self.intercept_ = float(intercepts[0])
        self.info_ = info
        return self


class SCADRegressor(_NonconvexRegressorBase):
    """Least-squares regression with SCAD penalty at a single λ."""

    def __init__(
        self,
        lambda_: float = 0.1,
        a: float = 3.7,
        *,
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        screening: str = "strong",
        acceleration: int | None = 5,
    ) -> None:
        self.lambda_ = lambda_
        self.a = a
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.screening = screening
        self.acceleration = acceleration

    def fit(self, x, y: NDArray[np.float64]) -> "SCADRegressor":
        w = (
            np.ascontiguousarray(self.weights, dtype=np.float64)
            if self.weights is not None
            else None
        )
        if _is_sparse(x):
            y = np.ascontiguousarray(y, dtype=np.float64)
            data, indices, indptr, n_rows, n_cols = _as_csc_arrays(x)
            if y.ndim != 1 or y.shape[0] != n_rows:
                raise ValueError(
                    f"y must be 1D with length {n_rows}, got shape {y.shape}"
                )
            coefs, intercepts, _, info = _core.solve_scad_ls_path_sparse(
                data, indices, indptr, n_rows, n_cols, y,
                a=self.a,
                lambdas=np.array([self.lambda_], dtype=np.float64),
                weights=w,
                max_iter=self.max_iter,
                tol=self.tol,
                screening=self.screening,
                acceleration=self.acceleration,
                fit_intercept=self.fit_intercept,
                standardize_x=self.standardize,
            )
            self.n_features_in_ = n_cols
        else:
            x, y = self._validate_xy(x, y)
            coefs, intercepts, _, info = _core.solve_scad_ls_path(
                x,
                y,
                a=self.a,
                lambdas=np.array([self.lambda_], dtype=np.float64),
                weights=w,
                max_iter=self.max_iter,
                tol=self.tol,
                screening=self.screening,
                acceleration=self.acceleration,
                fit_intercept=self.fit_intercept,
                standardize_x=self.standardize,
            )
            self.n_features_in_ = x.shape[1]
        self.coef_ = coefs[0]
        self.intercept_ = float(intercepts[0])
        self.info_ = info
        return self


class ElasticNetRegressor(_NonconvexRegressorBase):
    """Elastic-net least squares at a single λ.

    Convex penalty `α λ |β_j| + (1-α) λ β_j² / 2` per feature. ``alpha
    = 1`` recovers pure lasso (≈ ``MCPRegressor`` at large γ);
    ``alpha = 0`` recovers pure ridge. Matches glmnet's
    ``cv.glmnet(family="gaussian", alpha=...)`` shape.

    Parameters
    ----------
    lambda_ : float, default 0.1
    alpha : float, default 0.5
        Convex combination weight: ``α=1`` is lasso, ``α=0`` is ridge,
        intermediate is elastic net. Must be in [0, 1].
    weights : array-like of shape (n_features,) or None
        Per-feature penalty weights; apply to both L1 and L2 parts.
    max_iter : int, default 100
    tol : float, default 1e-6
    fit_intercept : bool, default True
    standardize : bool, default False
    screening : {"off", "strong", "gap_safe"}, default "strong"
    acceleration : int or None, default 5

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
    intercept_ : float
    info_ : dict
    n_features_in_ : int

    See Also
    --------
    skein_glm.ElasticNetPathRegressor : Full λ-path with warm starts.
    skein_glm.MCPRegressor : Nonconvex sparse alternative.
    """

    def __init__(
        self,
        lambda_: float = 0.1,
        alpha: float = 0.5,
        *,
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        screening: str = "strong",
        acceleration: int | None = 5,
    ) -> None:
        self.lambda_ = lambda_
        self.alpha = alpha
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.screening = screening
        self.acceleration = acceleration

    def fit(self, x, y: NDArray[np.float64]) -> "ElasticNetRegressor":
        w = (
            np.ascontiguousarray(self.weights, dtype=np.float64)
            if self.weights is not None
            else None
        )
        if _is_sparse(x):
            y = np.ascontiguousarray(y, dtype=np.float64)
            data, indices, indptr, n_rows, n_cols = _as_csc_arrays(x)
            if y.ndim != 1 or y.shape[0] != n_rows:
                raise ValueError(
                    f"y must be 1D with length {n_rows}, got shape {y.shape}"
                )
            coefs, intercepts, _, info = _core.solve_elastic_net_ls_path_sparse(
                data, indices, indptr, n_rows, n_cols, y,
                alpha=self.alpha,
                lambdas=np.array([self.lambda_], dtype=np.float64),
                weights=w,
                max_iter=self.max_iter,
                tol=self.tol,
                screening=self.screening,
                acceleration=self.acceleration,
                fit_intercept=self.fit_intercept,
                standardize_x=self.standardize,
            )
            self.n_features_in_ = n_cols
        else:
            x, y = self._validate_xy(x, y)
            coefs, intercepts, _, info = _core.solve_elastic_net_ls_path(
                x, y,
                alpha=self.alpha,
                lambdas=np.array([self.lambda_], dtype=np.float64),
                weights=w,
                max_iter=self.max_iter,
                tol=self.tol,
                screening=self.screening,
                acceleration=self.acceleration,
                fit_intercept=self.fit_intercept,
                standardize_x=self.standardize,
            )
            self.n_features_in_ = x.shape[1]
        self.coef_ = coefs[0]
        self.intercept_ = float(intercepts[0])
        self.info_ = info
        return self


class BridgeRegressor(_NonconvexRegressorBase):
    """Bridge (ℓ_q) regression at a single λ via outer LLA.

    Penalty `λ · Σ_j w_j |β_j|^q` with `q ∈ (0, 1]`. At ``q = 1`` this
    is plain weighted lasso; for ``q < 1`` the penalty is concave and
    sparsifies more aggressively than lasso (closes parity with
    ``grpreg``'s ``family="gaussian"`` + bridge alternative). The
    inner LLA surrogate is weighted lasso with per-coordinate weights
    `q · (|β_old| + ε)^(q-1) · w_base`.

    Parameters
    ----------
    lambda_ : float, default 0.1
    q : float, default 0.5
        Bridge exponent in (0, 1]. q = 0.5 is the canonical bridge
        choice in the literature.
    eps : float, default 1e-6
        Floor on `|β_j|` inside the LLA weight to keep the surrogate
        weight finite at β = 0. Smaller eps sharpens sparsification
        but increases outer LLA iterations.
    weights : array-like of shape (n_features,) or None
    max_iter : int, default 100
        Inner CD iterations per outer LLA step.
    tol : float, default 1e-6
    max_outer : int, default 10
        Maximum LLA outer iterations.
    outer_tol : float, default 1e-6
        Stop the LLA loop when `max_j |β_new − β_old| < outer_tol`.
    fit_intercept : bool, default True
    standardize : bool, default False
    acceleration : int or None, default 5

    See Also
    --------
    skein_glm.BridgePathRegressor : Full λ-path with warm starts.
    skein_glm.MCPRegressor : Closed-form-prox nonconvex alternative.
    """

    def __init__(
        self,
        lambda_: float = 0.1,
        q: float = 0.5,
        *,
        eps: float = 1e-6,
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
    ) -> None:
        self.lambda_ = lambda_
        self.q = q
        self.eps = eps
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.max_outer = max_outer
        self.outer_tol = outer_tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.acceleration = acceleration

    def fit(self, x, y: NDArray[np.float64]) -> "BridgeRegressor":
        w = (
            np.ascontiguousarray(self.weights, dtype=np.float64)
            if self.weights is not None
            else None
        )
        common: dict[str, Any] = dict(
            q=self.q,
            eps=self.eps,
            lambdas=np.array([self.lambda_], dtype=np.float64),
            weights=w,
            max_iter=self.max_iter,
            tol=self.tol,
            acceleration=self.acceleration,
            fit_intercept=self.fit_intercept,
            standardize_x=self.standardize,
            max_outer=self.max_outer,
            outer_tol=self.outer_tol,
        )
        if _is_sparse(x):
            y = np.ascontiguousarray(y, dtype=np.float64)
            data, indices, indptr, n_rows, n_cols = _as_csc_arrays(x)
            if y.ndim != 1 or y.shape[0] != n_rows:
                raise ValueError(
                    f"y must be 1D with length {n_rows}, got shape {y.shape}"
                )
            coefs, intercepts, _, info = _core.solve_bridge_ls_path_sparse(
                data, indices, indptr, n_rows, n_cols, y, **common,
            )
            self.n_features_in_ = n_cols
        else:
            x, y = self._validate_xy(x, y)
            coefs, intercepts, _, info = _core.solve_bridge_ls_path(
                x, y, **common,
            )
            self.n_features_in_ = x.shape[1]
        self.coef_ = coefs[0]
        self.intercept_ = float(intercepts[0])
        self.info_ = info
        return self


class _PathRegressorBase(BaseEstimator):
    """Common attributes for the path estimators."""

    coefs_: NDArray[np.float64]       # (n_lambdas, p) original-scale
    intercepts_: NDArray[np.float64]  # (n_lambdas,)
    lambdas_: NDArray[np.float64]     # (n_lambdas,) decreasing
    info_: dict[str, Any]
    n_features_in_: int

    def predict(self, x) -> NDArray[np.float64]:
        """Predictions for every λ in the path: returns (n_samples, n_lambdas)."""
        if _is_sparse(x):
            return np.asarray(x @ self.coefs_.T) + self.intercepts_[None, :]
        x = np.ascontiguousarray(x, dtype=np.float64)
        return x @ self.coefs_.T + self.intercepts_[None, :]


class MCPPathRegressor(_PathRegressorBase):
    """MCP-penalized least squares along a λ-path with warm starts.

    Solves

        min_β (1/2n) ‖y − Xβ − α‖² + Σ_j w_j · ρ_MCP(β_j; λ, γ)

    for a sequence of λ values, threading β across decreasing λ as
    warm-starts. Equivalent to lasso when ``gamma`` is large
    (``gamma=1e6`` is a good convex stand-in); aggressively
    sparsifies as ``gamma`` shrinks.

    Accepts numpy arrays, ``scipy.sparse.csc_matrix``,
    :class:`skein_glm.MmapDesignF64` / :class:`skein_glm.MmapDesignF32`,
    and :class:`skein_glm.ChunkedDesignF64` / :class:`skein_glm.ChunkedDesignF32`
    transparently.

    Parameters
    ----------
    gamma : float, default 3.0
        MCP nonconvexity parameter (>1). Smaller is more aggressive;
        ``gamma=3`` matches ``ncvreg``'s default. ``gamma=1e6`` is
        numerically convex (≈ lasso).
    lambdas : array-like or None, default None
        Explicit λ grid (descending). If None, derived from λ_max
        (the smallest λ that gives β = 0 by KKT) and a geometric
        grid of length ``n_lambdas`` with ratio ``lambda_min_ratio``.
    n_lambdas : int, default 100
        Length of the auto-generated λ grid. Ignored if ``lambdas``
        is given.
    lambda_min_ratio : float, default 1e-3
        Smallest λ in the auto-grid as a fraction of λ_max.
    weights : array-like of shape (n_features,) or None, default None
        Per-feature penalty weights (the ``w_j`` above). ``None``
        means uniform weights of 1.
    max_iter : int, default 100
        Maximum number of CD iterations per λ.
    tol : float, default 1e-6
        Convergence threshold (max coordinate-update L1 in
        coefficient space).
    fit_intercept : bool, default True
        If True, an unpenalized intercept is fit alongside β.
    standardize : bool, default False
        If True, columns of X are scaled to unit variance before
        fitting; ``coef_`` / ``intercept_`` are returned in
        original-feature scale.
    screening : {"off", "strong", "gap_safe"}, default "strong"
        Working-set strategy. "strong" = Tibshirani sequential strong
        rule + KKT verification. "gap_safe" = Fercoq-Gramfort-Salmon
        sphere screening. "off" disables screening.
    acceleration : int or None, default 5
        Anderson acceleration depth on the iterate sequence. ``None``
        disables acceleration.

    Attributes
    ----------
    coefs_ : ndarray of shape (n_lambdas, n_features)
        Fitted coefficients per λ. Each row is the β at the
        corresponding ``lambdas_[k]``.
    intercepts_ : ndarray of shape (n_lambdas,)
        Fitted intercepts per λ.
    lambdas_ : ndarray of shape (n_lambdas,)
        The λ values actually used (descending).
    info_ : dict
        Solver diagnostics: per-λ iteration counts, convergence
        flags, final objective values, working-set sizes, KKT-pass
        counts.
    n_features_in_ : int
        Number of features in ``X``.

    See Also
    --------
    skein_glm.MCPRegressor : Single-λ version.
    skein_glm.MCPPathCV : K-fold cross-validated path with auto λ-selection.
    skein_glm.SCADPathRegressor : SCAD-penalized analogue.

    Examples
    --------
    >>> import numpy as np
    >>> import skein_glm
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((200, 50))
    >>> y = X[:, :3] @ [1.5, -2.0, 0.8] + 0.1 * rng.standard_normal(200)
    >>> model = skein_glm.MCPPathRegressor(gamma=3.0, n_lambdas=50).fit(X, y)
    >>> model.coefs_.shape
    (50, 50)
    """

    def __init__(
        self,
        gamma: float = 3.0,
        *,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        weights: NDArray[np.float64] | None = None,
        sample_weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        screening: str = "strong",
        acceleration: int | None = 5,
    ) -> None:
        self.gamma = gamma
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.weights = weights
        self.sample_weights = sample_weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.screening = screening
        self.acceleration = acceleration

    def fit(self, x, y: NDArray[np.float64]) -> "MCPPathRegressor":
        y = np.ascontiguousarray(y, dtype=np.float64)
        sw = _validate_sample_weights(self.sample_weights, x)
        _reject_sample_weights_on_non_dense(sw, x, estimator_kind="MCPPathRegressor")
        lams = (
            np.ascontiguousarray(self.lambdas, dtype=np.float64)
            if self.lambdas is not None
            else None
        )
        w = (
            np.ascontiguousarray(self.weights, dtype=np.float64)
            if self.weights is not None
            else None
        )
        if _is_chunked(x):
            if y.ndim != 1 or y.shape[0] != x.n_rows:
                raise ValueError(
                    f"y must be 1D with length {x.n_rows}, got shape {y.shape}"
                )
            chunked_entry = (
                _core.solve_mcp_ls_path_chunked_f32
                if x.dtype == "f32"
                else _core.solve_mcp_ls_path_chunked
            )
            coefs, intercepts, lambdas_used, info = chunked_entry(
                x.chunks, x.n_cols, y,
                gamma=self.gamma,
                lambdas=lams,
                n_lambdas=self.n_lambdas,
                lambda_min_ratio=self.lambda_min_ratio,
                weights=w,
                max_iter=self.max_iter,
                tol=self.tol,
                screening=self.screening,
                acceleration=self.acceleration,
                fit_intercept=self.fit_intercept,
                standardize_x=self.standardize,
            )
            self.n_features_in_ = x.n_cols
        elif _is_mmap(x):
            if y.ndim != 1 or y.shape[0] != x.n_rows:
                raise ValueError(
                    f"y must be 1D with length {x.n_rows}, got shape {y.shape}"
                )
            mmap_entry = (
                _core.solve_mcp_ls_path_mmap_f32
                if x.dtype == "f32"
                else _core.solve_mcp_ls_path_mmap
            )
            coefs, intercepts, lambdas_used, info = mmap_entry(
                x.path, x.n_rows, x.n_cols, y,
                gamma=self.gamma,
                lambdas=lams,
                n_lambdas=self.n_lambdas,
                lambda_min_ratio=self.lambda_min_ratio,
                weights=w,
                max_iter=self.max_iter,
                tol=self.tol,
                screening=self.screening,
                acceleration=self.acceleration,
                fit_intercept=self.fit_intercept,
                standardize_x=self.standardize,
            )
            self.n_features_in_ = x.n_cols
        elif _is_sparse(x):
            data, indices, indptr, n_rows, n_cols = _as_csc_arrays(x)
            if y.ndim != 1 or y.shape[0] != n_rows:
                raise ValueError(
                    f"y must be 1D with length {n_rows}, got shape {y.shape}"
                )
            coefs, intercepts, lambdas_used, info = _core.solve_mcp_ls_path_sparse(
                data, indices, indptr, n_rows, n_cols, y,
                gamma=self.gamma,
                lambdas=lams,
                n_lambdas=self.n_lambdas,
                lambda_min_ratio=self.lambda_min_ratio,
                weights=w,
                max_iter=self.max_iter,
                tol=self.tol,
                screening=self.screening,
                acceleration=self.acceleration,
                fit_intercept=self.fit_intercept,
                standardize_x=self.standardize,
            )
            self.n_features_in_ = n_cols
        else:
            x = np.ascontiguousarray(x, dtype=np.float64)
            coefs, intercepts, lambdas_used, info = _core.solve_mcp_ls_path(
                x,
                y,
                gamma=self.gamma,
                lambdas=lams,
                n_lambdas=self.n_lambdas,
                lambda_min_ratio=self.lambda_min_ratio,
                weights=w,
                sample_weights=sw,
                max_iter=self.max_iter,
                tol=self.tol,
                screening=self.screening,
                acceleration=self.acceleration,
                fit_intercept=self.fit_intercept,
                standardize_x=self.standardize,
            )
            self.n_features_in_ = x.shape[1]
        self.coefs_ = coefs
        self.intercepts_ = intercepts
        self.lambdas_ = lambdas_used
        self.info_ = info
        return self


class SCADPathRegressor(_PathRegressorBase):
    """SCAD regression along an entire λ-path with warm starts."""

    def __init__(
        self,
        a: float = 3.7,
        *,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        weights: NDArray[np.float64] | None = None,
        sample_weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        screening: str = "strong",
        acceleration: int | None = 5,
    ) -> None:
        self.a = a
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.weights = weights
        self.sample_weights = sample_weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.screening = screening
        self.acceleration = acceleration

    def fit(self, x, y: NDArray[np.float64]) -> "SCADPathRegressor":
        y = np.ascontiguousarray(y, dtype=np.float64)
        sw = _validate_sample_weights(self.sample_weights, x)
        _reject_sample_weights_on_non_dense(sw, x, estimator_kind="SCADPathRegressor")
        lams = (
            np.ascontiguousarray(self.lambdas, dtype=np.float64)
            if self.lambdas is not None
            else None
        )
        w = (
            np.ascontiguousarray(self.weights, dtype=np.float64)
            if self.weights is not None
            else None
        )
        if _is_sparse(x):
            data, indices, indptr, n_rows, n_cols = _as_csc_arrays(x)
            if y.ndim != 1 or y.shape[0] != n_rows:
                raise ValueError(
                    f"y must be 1D with length {n_rows}, got shape {y.shape}"
                )
            coefs, intercepts, lambdas_used, info = _core.solve_scad_ls_path_sparse(
                data, indices, indptr, n_rows, n_cols, y,
                a=self.a,
                lambdas=lams,
                n_lambdas=self.n_lambdas,
                lambda_min_ratio=self.lambda_min_ratio,
                weights=w,
                max_iter=self.max_iter,
                tol=self.tol,
                screening=self.screening,
                acceleration=self.acceleration,
                fit_intercept=self.fit_intercept,
                standardize_x=self.standardize,
            )
            self.n_features_in_ = n_cols
        else:
            x = np.ascontiguousarray(x, dtype=np.float64)
            coefs, intercepts, lambdas_used, info = _core.solve_scad_ls_path(
                x,
                y,
                a=self.a,
                lambdas=lams,
                n_lambdas=self.n_lambdas,
                lambda_min_ratio=self.lambda_min_ratio,
                weights=w,
                sample_weights=sw,
                max_iter=self.max_iter,
                tol=self.tol,
                screening=self.screening,
                acceleration=self.acceleration,
                fit_intercept=self.fit_intercept,
                standardize_x=self.standardize,
            )
            self.n_features_in_ = x.shape[1]
        self.coefs_ = coefs
        self.intercepts_ = intercepts
        self.lambdas_ = lambdas_used
        self.info_ = info
        return self


class ElasticNetPathRegressor(_PathRegressorBase):
    """Elastic-net least squares along a λ-path with warm starts.

    Solves ``min_β (1/2n) ‖y − Xβ − α_int‖² + Σ_j w_j λ [α |β_j| + (1-α) β_j² / 2]``
    for a sequence of λ values, threading β across decreasing λ as
    warm-starts. Convex objective, so coordinate descent converges
    to the global optimum at every λ.

    Recovers pure lasso at ``alpha=1`` (≈ ``MCPPathRegressor`` at
    ``gamma=1e6``) and pure ridge at ``alpha=0``. Matches glmnet's
    ``glmnet(family="gaussian", alpha=...)`` shape.

    Parameters
    ----------
    alpha : float, default 0.5
        Convex combination weight in [0, 1]. ``alpha=1`` is lasso;
        ``alpha=0`` is ridge.
    lambdas : array-like or None, default None
        Explicit λ grid (descending). If None, derived from λ_max
        (computed against the L1-effective weights ``α · w_j``) and
        a geometric grid.
    n_lambdas : int, default 100
    lambda_min_ratio : float, default 1e-3
    weights : array-like of shape (n_features,) or None
        Per-feature penalty weights (apply to both L1 and L2 parts).
    max_iter : int, default 100
    tol : float, default 1e-6
    fit_intercept : bool, default True
    standardize : bool, default False
    screening : {"off", "strong", "gap_safe"}, default "strong"
    acceleration : int or None, default 5

    Attributes
    ----------
    coefs_ : ndarray of shape (n_lambdas, n_features)
    intercepts_ : ndarray of shape (n_lambdas,)
    lambdas_ : ndarray of shape (n_lambdas,)
    info_ : dict
    n_features_in_ : int

    See Also
    --------
    skein_glm.ElasticNetRegressor : Single-λ version.
    skein_glm.ElasticNetPathCV : K-fold cross-validated path with auto λ-selection.
    skein_glm.MCPPathRegressor : Nonconvex sparse alternative.

    Examples
    --------
    >>> import numpy as np
    >>> import skein_glm
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((200, 50))
    >>> y = X[:, :3] @ [1.5, -2.0, 0.8] + 0.1 * rng.standard_normal(200)
    >>> model = skein_glm.ElasticNetPathRegressor(alpha=0.5, n_lambdas=50).fit(X, y)
    >>> model.coefs_.shape
    (50, 50)
    """

    def __init__(
        self,
        alpha: float = 0.5,
        *,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        weights: NDArray[np.float64] | None = None,
        sample_weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        screening: str = "strong",
        acceleration: int | None = 5,
    ) -> None:
        self.alpha = alpha
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.weights = weights
        self.sample_weights = sample_weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.screening = screening
        self.acceleration = acceleration

    def fit(self, x, y: NDArray[np.float64]) -> "ElasticNetPathRegressor":
        y = np.ascontiguousarray(y, dtype=np.float64)
        sw = _validate_sample_weights(self.sample_weights, x)
        _reject_sample_weights_on_non_dense(sw, x, estimator_kind="ElasticNetPathRegressor")
        lams = (
            np.ascontiguousarray(self.lambdas, dtype=np.float64)
            if self.lambdas is not None
            else None
        )
        w = (
            np.ascontiguousarray(self.weights, dtype=np.float64)
            if self.weights is not None
            else None
        )
        if _is_sparse(x):
            data, indices, indptr, n_rows, n_cols = _as_csc_arrays(x)
            if y.ndim != 1 or y.shape[0] != n_rows:
                raise ValueError(
                    f"y must be 1D with length {n_rows}, got shape {y.shape}"
                )
            coefs, intercepts, lambdas_used, info = _core.solve_elastic_net_ls_path_sparse(
                data, indices, indptr, n_rows, n_cols, y,
                alpha=self.alpha,
                lambdas=lams,
                n_lambdas=self.n_lambdas,
                lambda_min_ratio=self.lambda_min_ratio,
                weights=w,
                max_iter=self.max_iter,
                tol=self.tol,
                screening=self.screening,
                acceleration=self.acceleration,
                fit_intercept=self.fit_intercept,
                standardize_x=self.standardize,
            )
            self.n_features_in_ = n_cols
        else:
            x = np.ascontiguousarray(x, dtype=np.float64)
            coefs, intercepts, lambdas_used, info = _core.solve_elastic_net_ls_path(
                x, y,
                alpha=self.alpha,
                lambdas=lams,
                n_lambdas=self.n_lambdas,
                lambda_min_ratio=self.lambda_min_ratio,
                weights=w,
                sample_weights=sw,
                max_iter=self.max_iter,
                tol=self.tol,
                screening=self.screening,
                acceleration=self.acceleration,
                fit_intercept=self.fit_intercept,
                standardize_x=self.standardize,
            )
            self.n_features_in_ = x.shape[1]
        self.coefs_ = coefs
        self.intercepts_ = intercepts
        self.lambdas_ = lambdas_used
        self.info_ = info
        return self


class BridgePathRegressor(_PathRegressorBase):
    """Bridge (ℓ_q) regression along a λ-path with warm starts.

    Penalty `λ · Σ_j w_j |β_j|^q` with `q ∈ (0, 1]`, fit via outer LLA
    wrapping a weighted-lasso inner. Warm-starts β across decreasing λ;
    a fresh LLA outer loop runs at each λ.

    Parameters
    ----------
    q : float, default 0.5
        Bridge exponent in (0, 1]. ``q = 1`` is plain weighted lasso;
        smaller q sparsifies more aggressively but the loss surface
        becomes more non-convex.
    eps : float, default 1e-6
    lambdas : array-like or None, default None
    n_lambdas : int, default 100
    lambda_min_ratio : float, default 1e-3
    weights : array-like of shape (n_features,) or None
    max_iter : int, default 100
    tol : float, default 1e-6
    max_outer : int, default 10
    outer_tol : float, default 1e-6
    fit_intercept : bool, default True
    standardize : bool, default False
    acceleration : int or None, default 5

    Attributes
    ----------
    coefs_ : ndarray of shape (n_lambdas, n_features)
    intercepts_ : ndarray of shape (n_lambdas,)
    lambdas_ : ndarray of shape (n_lambdas,)
    info_ : dict
        ``outer_iters``, ``outer_converged``, ``inner_iters``,
        ``final_objs``.
    n_features_in_ : int

    See Also
    --------
    skein_glm.BridgeRegressor : Single-λ version.
    skein_glm.BridgePathCV : K-fold cross-validated path.
    """

    def __init__(
        self,
        q: float = 0.5,
        *,
        eps: float = 1e-6,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
    ) -> None:
        self.q = q
        self.eps = eps
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.max_outer = max_outer
        self.outer_tol = outer_tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.acceleration = acceleration

    def fit(self, x, y: NDArray[np.float64]) -> "BridgePathRegressor":
        y = np.ascontiguousarray(y, dtype=np.float64)
        lams = (
            np.ascontiguousarray(self.lambdas, dtype=np.float64)
            if self.lambdas is not None
            else None
        )
        w = (
            np.ascontiguousarray(self.weights, dtype=np.float64)
            if self.weights is not None
            else None
        )
        common: dict[str, Any] = dict(
            q=self.q,
            eps=self.eps,
            lambdas=lams,
            n_lambdas=self.n_lambdas,
            lambda_min_ratio=self.lambda_min_ratio,
            weights=w,
            max_iter=self.max_iter,
            tol=self.tol,
            acceleration=self.acceleration,
            fit_intercept=self.fit_intercept,
            standardize_x=self.standardize,
            max_outer=self.max_outer,
            outer_tol=self.outer_tol,
        )
        if _is_sparse(x):
            data, indices, indptr, n_rows, n_cols = _as_csc_arrays(x)
            if y.ndim != 1 or y.shape[0] != n_rows:
                raise ValueError(
                    f"y must be 1D with length {n_rows}, got shape {y.shape}"
                )
            coefs, intercepts, lambdas_used, info = _core.solve_bridge_ls_path_sparse(
                data, indices, indptr, n_rows, n_cols, y, **common,
            )
            self.n_features_in_ = n_cols
        else:
            x = np.ascontiguousarray(x, dtype=np.float64)
            coefs, intercepts, lambdas_used, info = _core.solve_bridge_ls_path(
                x, y, **common,
            )
            self.n_features_in_ = x.shape[1]
        self.coefs_ = coefs
        self.intercepts_ = intercepts
        self.lambdas_ = lambdas_used
        self.info_ = info
        return self


# =====================================================================
# Group estimators
# =====================================================================


class _GroupEstimatorMixin:
    """Shared validation: groups labels become an int64 numpy array."""

    def _validate_groups(self, groups, n_features: int) -> NDArray[np.int64]:
        groups_arr = np.asarray(groups, dtype=np.int64)
        if groups_arr.ndim != 1 or groups_arr.shape[0] != n_features:
            raise ValueError(
                f"groups must be 1D of length n_features={n_features}, "
                f"got shape {groups_arr.shape}"
            )
        return np.ascontiguousarray(groups_arr)


class _GroupSingleLambdaBase(_NonconvexRegressorBase, _GroupEstimatorMixin):
    """Common base for single-λ group regressors."""

    pass


class _GroupPathBase(_PathRegressorBase, _GroupEstimatorMixin):
    """Common base for full-path group regressors."""

    pass


def _glm_dispatch_inputs(
    estimator,
    x,
    y,
    *,
    validate_y,
    is_path: bool,
    groups=None,
):
    """Common dense/sparse pre-processing for GLM scalar and group
    estimators (logistic, Poisson). `groups` is `None` for scalar."""
    w = (
        np.ascontiguousarray(estimator.weights, dtype=np.float64)
        if estimator.weights is not None
        else None
    )
    if is_path:
        lams = (
            np.ascontiguousarray(estimator.lambdas, dtype=np.float64)
            if estimator.lambdas is not None
            else None
        )
    else:
        lams = np.array([estimator.lambda_], dtype=np.float64)

    common: dict[str, Any] = dict(
        lambdas=lams,
        weights=w,
        max_iter=estimator.max_iter,
        tol=estimator.tol,
        acceleration=estimator.acceleration,
        fit_intercept=estimator.fit_intercept,
        standardize_x=estimator.standardize,
        max_outer=estimator.max_outer,
        outer_tol=estimator.outer_tol,
    )
    if is_path:
        common["n_lambdas"] = estimator.n_lambdas
        common["lambda_min_ratio"] = estimator.lambda_min_ratio

    # Poisson estimators support an `offset` argument (log-exposure for
    # rate models). Logistic / Cox don't define `offset` so this is a
    # no-op for them via getattr's default.
    offset = getattr(estimator, "offset", None)
    if offset is not None:
        common["offset"] = np.ascontiguousarray(offset, dtype=np.float64)

    # Per-sample weights — opt-in per estimator. Validated against the
    # design's row count; only forwarded into the `_core` call when
    # actually set, since most estimators won't accept the kwarg yet.
    sample_weights = getattr(estimator, "sample_weights", None)
    sw_arr = _validate_sample_weights(sample_weights, x)
    if sw_arr is not None:
        _reject_sample_weights_on_non_dense(
            sw_arr, x, estimator_kind=type(estimator).__name__
        )
        common["sample_weights"] = sw_arr

    if _is_sparse(x):
        y_arr = np.ascontiguousarray(y, dtype=np.float64)
        data, indices, indptr, n_rows, n_cols = _as_csc_arrays(x)
        if y_arr.ndim != 1 or y_arr.shape[0] != n_rows:
            raise ValueError(
                f"y must be 1D with length {n_rows}, got shape {y_arr.shape}"
            )
        validate_y(y_arr)
        if groups is not None:
            groups_arr = estimator._validate_groups(groups, n_cols)
            return common, (data, indices, indptr, n_rows, n_cols, y_arr, groups_arr), n_cols
        return common, (data, indices, indptr, n_rows, n_cols, y_arr), n_cols

    x_arr = np.ascontiguousarray(x, dtype=np.float64)
    y_arr = np.ascontiguousarray(y, dtype=np.float64)
    if x_arr.ndim != 2:
        raise ValueError(f"x must be 2D, got shape {x_arr.shape}")
    if y_arr.ndim != 1 or y_arr.shape[0] != x_arr.shape[0]:
        raise ValueError(
            f"y must be 1D with length {x_arr.shape[0]}, got shape {y_arr.shape}"
        )
    validate_y(y_arr)
    common["_x"] = x_arr
    common["_y"] = y_arr
    if groups is not None:
        common["_groups"] = estimator._validate_groups(groups, x_arr.shape[1])
    return common, None, x_arr.shape[1]


def _cox_dispatch_inputs(
    estimator,
    x,
    time,
    event,
    *,
    is_path: bool,
    groups=None,
):
    """Common dense/sparse pre-processing for Cox PH estimators
    (no `fit_intercept`)."""
    w = (
        np.ascontiguousarray(estimator.weights, dtype=np.float64)
        if estimator.weights is not None
        else None
    )
    if is_path:
        lams = (
            np.ascontiguousarray(estimator.lambdas, dtype=np.float64)
            if estimator.lambdas is not None
            else None
        )
    else:
        lams = np.array([estimator.lambda_], dtype=np.float64)

    common: dict[str, Any] = dict(
        lambdas=lams,
        weights=w,
        max_iter=estimator.max_iter,
        tol=estimator.tol,
        acceleration=estimator.acceleration,
        standardize_x=estimator.standardize,
        max_outer=estimator.max_outer,
        outer_tol=estimator.outer_tol,
    )
    if is_path:
        common["n_lambdas"] = estimator.n_lambdas
        common["lambda_min_ratio"] = estimator.lambda_min_ratio

    time_arr = np.ascontiguousarray(time, dtype=np.float64)
    event_arr = np.ascontiguousarray(event, dtype=np.float64)
    if not np.all(np.isfinite(time_arr)) or np.any(time_arr < 0.0):
        raise ValueError("Cox PH requires time ≥ 0 (finite)")
    if not np.all((event_arr == 0.0) | (event_arr == 1.0)):
        raise ValueError("Cox PH requires event ∈ {0, 1}")
    if event_arr.sum() < 1:
        raise ValueError("Cox PH requires at least one event (event = 1)")

    if _is_sparse(x):
        data, indices, indptr, n_rows, n_cols = _as_csc_arrays(x)
        if time_arr.shape != (n_rows,) or event_arr.shape != (n_rows,):
            raise ValueError(
                f"time and event must each be 1D with length {n_rows}"
            )
        if groups is not None:
            groups_arr = estimator._validate_groups(groups, n_cols)
            return (
                common,
                (data, indices, indptr, n_rows, n_cols, time_arr, event_arr, groups_arr),
                n_cols,
            )
        return common, (data, indices, indptr, n_rows, n_cols, time_arr, event_arr), n_cols

    x_arr = np.ascontiguousarray(x, dtype=np.float64)
    if x_arr.ndim != 2:
        raise ValueError(f"x must be 2D, got shape {x_arr.shape}")
    if time_arr.shape != (x_arr.shape[0],) or event_arr.shape != (x_arr.shape[0],):
        raise ValueError(
            f"time and event must each be 1D with length {x_arr.shape[0]}"
        )
    common["_x"] = x_arr
    common["_time"] = time_arr
    common["_event"] = event_arr
    if groups is not None:
        common["_groups"] = estimator._validate_groups(groups, x_arr.shape[1])
    return common, None, x_arr.shape[1]


def _ls_group_dispatch_inputs(
    estimator,
    x,
    y,
    groups,
    *,
    is_path: bool,
):
    """Common dense/sparse pre-processing for LS group estimators.

    Returns `(common_kwargs, sparse_payload, n_features)` where
    `sparse_payload` is `None` for dense (use `(x, y)` from
    `common_kwargs['_x']` / `['_y']`) or a tuple
    `(data, indices, indptr, n_rows, n_cols, y, groups_arr)` for
    sparse. The caller then builds the correct PyO3 call site."""
    w = (
        np.ascontiguousarray(estimator.weights, dtype=np.float64)
        if estimator.weights is not None
        else None
    )
    if is_path:
        lams = (
            np.ascontiguousarray(estimator.lambdas, dtype=np.float64)
            if estimator.lambdas is not None
            else None
        )
    else:
        lams = np.array([estimator.lambda_], dtype=np.float64)

    common: dict[str, Any] = dict(
        lambdas=lams,
        weights=w,
        max_iter=estimator.max_iter,
        tol=estimator.tol,
        screening=estimator.screening,
        acceleration=estimator.acceleration,
        parallel=estimator.parallel,
        fit_intercept=estimator.fit_intercept,
    )
    if is_path:
        common["n_lambdas"] = estimator.n_lambdas
        common["lambda_min_ratio"] = estimator.lambda_min_ratio

    if _is_sparse(x):
        y_arr = np.ascontiguousarray(y, dtype=np.float64)
        data, indices, indptr, n_rows, n_cols = _as_csc_arrays(x)
        if y_arr.ndim != 1 or y_arr.shape[0] != n_rows:
            raise ValueError(
                f"y must be 1D with length {n_rows}, got shape {y_arr.shape}"
            )
        groups_arr = estimator._validate_groups(groups, n_cols)
        # Sparse path takes `standardize_x` directly (lazy Standardized
        # wrapper applied internally).
        common["standardize_x"] = estimator.standardize
        return common, (data, indices, indptr, n_rows, n_cols, y_arr, groups_arr), n_cols

    x_arr = np.ascontiguousarray(x, dtype=np.float64)
    y_arr = np.ascontiguousarray(y, dtype=np.float64)
    if x_arr.ndim != 2:
        raise ValueError(f"x must be 2D, got shape {x_arr.shape}")
    if y_arr.ndim != 1 or y_arr.shape[0] != x_arr.shape[0]:
        raise ValueError(
            f"y must be 1D with length {x_arr.shape[0]}, got shape {y_arr.shape}"
        )
    groups_arr = estimator._validate_groups(groups, x_arr.shape[1])
    common["standardize_x"] = estimator.standardize
    common["_x"] = x_arr
    common["_y"] = y_arr
    common["_groups"] = groups_arr
    return common, None, x_arr.shape[1]


# ---- group lasso (convex) ------------------------------------------------


class GroupLassoRegressor(_GroupSingleLambdaBase):
    """Group lasso at a single λ."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        lambda_: float = 0.1,
        *,
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        screening: str = "strong",
        acceleration: int | None = 5,
        parallel: bool = False,
    ) -> None:
        self.groups = groups
        self.lambda_ = lambda_
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.screening = screening
        self.acceleration = acceleration
        self.parallel = parallel

    def fit(self, x, y: NDArray[np.float64]) -> "GroupLassoRegressor":
        common, sparse_payload, n_features = _ls_group_dispatch_inputs(
            self, x, y, self.groups, is_path=False,
        )
        if sparse_payload is not None:
            data, indices, indptr, n_rows, n_cols, y_arr, groups_arr = sparse_payload
            coefs, intercepts, _, info = _core.solve_group_lasso_ls_path_sparse(
                data, indices, indptr, n_rows, n_cols, y_arr, groups_arr, **common,
            )
        else:
            x_arr = common.pop("_x")
            y_arr = common.pop("_y")
            groups_arr = common.pop("_groups")
            coefs, intercepts, _, info = _core.solve_group_lasso_ls_path(
                x_arr, y_arr, groups_arr, **common,
            )
        self.coef_ = coefs[0]
        self.intercept_ = float(intercepts[0])
        self.info_ = info
        self.n_features_in_ = n_features
        return self


class GroupLassoPathRegressor(_GroupPathBase):
    """Group lasso along an entire λ-path with warm starts."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        *,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        screening: str = "strong",
        acceleration: int | None = 5,
        parallel: bool = False,
    ) -> None:
        self.groups = groups
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.screening = screening
        self.acceleration = acceleration
        self.parallel = parallel

    def fit(self, x, y: NDArray[np.float64]) -> "GroupLassoPathRegressor":
        common, sparse_payload, n_features = _ls_group_dispatch_inputs(
            self, x, y, self.groups, is_path=True,
        )
        if sparse_payload is not None:
            data, indices, indptr, n_rows, n_cols, y_arr, groups_arr = sparse_payload
            coefs, intercepts, lambdas_used, info = _core.solve_group_lasso_ls_path_sparse(
                data, indices, indptr, n_rows, n_cols, y_arr, groups_arr, **common,
            )
        else:
            x_arr = common.pop("_x")
            y_arr = common.pop("_y")
            groups_arr = common.pop("_groups")
            coefs, intercepts, lambdas_used, info = _core.solve_group_lasso_ls_path(
                x_arr, y_arr, groups_arr, **common,
            )
        self.coefs_ = coefs
        self.intercepts_ = intercepts
        self.lambdas_ = lambdas_used
        self.info_ = info
        self.n_features_in_ = n_features
        return self


# ---- group MCP (LLA-wrapped) --------------------------------------------


class GroupMCPRegressor(_GroupSingleLambdaBase):
    """Group MCP at a single λ, solved by LLA outer loop."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        lambda_: float = 0.1,
        gamma: float = 3.0,
        *,
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        screening: str = "strong",
        acceleration: int | None = 5,
        parallel: bool = False,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.groups = groups
        self.lambda_ = lambda_
        self.gamma = gamma
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.screening = screening
        self.acceleration = acceleration
        self.parallel = parallel
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def fit(self, x, y: NDArray[np.float64]) -> "GroupMCPRegressor":
        common, sparse_payload, n_features = _ls_group_dispatch_inputs(
            self, x, y, self.groups, is_path=False,
        )
        common["gamma"] = self.gamma
        common["max_outer"] = self.max_outer
        common["outer_tol"] = self.outer_tol
        if sparse_payload is not None:
            data, indices, indptr, n_rows, n_cols, y_arr, groups_arr = sparse_payload
            coefs, intercepts, _, info = _core.solve_group_mcp_ls_path_sparse(
                data, indices, indptr, n_rows, n_cols, y_arr, groups_arr, **common,
            )
        else:
            x_arr = common.pop("_x")
            y_arr = common.pop("_y")
            groups_arr = common.pop("_groups")
            coefs, intercepts, _, info = _core.solve_group_mcp_ls_path(
                x_arr, y_arr, groups_arr, **common,
            )
        self.coef_ = coefs[0]
        self.intercept_ = float(intercepts[0])
        self.info_ = info
        self.n_features_in_ = n_features
        return self


class GroupMCPPathRegressor(_GroupPathBase):
    """Group MCP along an entire λ-path; LLA at every λ."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        gamma: float = 3.0,
        *,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        screening: str = "strong",
        acceleration: int | None = 5,
        parallel: bool = False,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.groups = groups
        self.gamma = gamma
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.screening = screening
        self.acceleration = acceleration
        self.parallel = parallel
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def fit(self, x, y: NDArray[np.float64]) -> "GroupMCPPathRegressor":
        common, sparse_payload, n_features = _ls_group_dispatch_inputs(
            self, x, y, self.groups, is_path=True,
        )
        common["gamma"] = self.gamma
        common["max_outer"] = self.max_outer
        common["outer_tol"] = self.outer_tol
        if sparse_payload is not None:
            data, indices, indptr, n_rows, n_cols, y_arr, groups_arr = sparse_payload
            coefs, intercepts, lambdas_used, info = _core.solve_group_mcp_ls_path_sparse(
                data, indices, indptr, n_rows, n_cols, y_arr, groups_arr, **common,
            )
        else:
            x_arr = common.pop("_x")
            y_arr = common.pop("_y")
            groups_arr = common.pop("_groups")
            coefs, intercepts, lambdas_used, info = _core.solve_group_mcp_ls_path(
                x_arr, y_arr, groups_arr, **common,
            )
        self.coefs_ = coefs
        self.intercepts_ = intercepts
        self.lambdas_ = lambdas_used
        self.info_ = info
        self.n_features_in_ = n_features
        return self


class GroupSCADRegressor(_GroupSingleLambdaBase):
    """Group SCAD at a single λ, solved by LLA outer loop. SCAD shape
    `a > 2` (default 3.7 — Fan & Li's recommendation)."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        lambda_: float = 0.1,
        a: float = 3.7,
        *,
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        screening: str = "strong",
        acceleration: int | None = 5,
        parallel: bool = False,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.groups = groups
        self.lambda_ = lambda_
        self.a = a
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.screening = screening
        self.acceleration = acceleration
        self.parallel = parallel
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def fit(self, x, y: NDArray[np.float64]) -> "GroupSCADRegressor":
        common, sparse_payload, n_features = _ls_group_dispatch_inputs(
            self, x, y, self.groups, is_path=False,
        )
        common["a"] = self.a
        common["max_outer"] = self.max_outer
        common["outer_tol"] = self.outer_tol
        if sparse_payload is not None:
            data, indices, indptr, n_rows, n_cols, y_arr, groups_arr = sparse_payload
            coefs, intercepts, _, info = _core.solve_group_scad_ls_path_sparse(
                data, indices, indptr, n_rows, n_cols, y_arr, groups_arr, **common,
            )
        else:
            x_arr = common.pop("_x")
            y_arr = common.pop("_y")
            groups_arr = common.pop("_groups")
            coefs, intercepts, _, info = _core.solve_group_scad_ls_path(
                x_arr, y_arr, groups_arr, **common,
            )
        self.coef_ = coefs[0]
        self.intercept_ = float(intercepts[0])
        self.info_ = info
        self.n_features_in_ = n_features
        return self


class GroupSCADPathRegressor(_GroupPathBase):
    """Group SCAD along an entire λ-path; LLA at every λ. SCAD shape
    `a > 2` (default 3.7)."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        a: float = 3.7,
        *,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        screening: str = "strong",
        acceleration: int | None = 5,
        parallel: bool = False,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.groups = groups
        self.a = a
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.screening = screening
        self.acceleration = acceleration
        self.parallel = parallel
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def fit(self, x, y: NDArray[np.float64]) -> "GroupSCADPathRegressor":
        common, sparse_payload, n_features = _ls_group_dispatch_inputs(
            self, x, y, self.groups, is_path=True,
        )
        common["a"] = self.a
        common["max_outer"] = self.max_outer
        common["outer_tol"] = self.outer_tol
        if sparse_payload is not None:
            data, indices, indptr, n_rows, n_cols, y_arr, groups_arr = sparse_payload
            coefs, intercepts, lambdas_used, info = _core.solve_group_scad_ls_path_sparse(
                data, indices, indptr, n_rows, n_cols, y_arr, groups_arr, **common,
            )
        else:
            x_arr = common.pop("_x")
            y_arr = common.pop("_y")
            groups_arr = common.pop("_groups")
            coefs, intercepts, lambdas_used, info = _core.solve_group_scad_ls_path(
                x_arr, y_arr, groups_arr, **common,
            )
        self.coefs_ = coefs
        self.intercepts_ = intercepts
        self.lambdas_ = lambdas_used
        self.info_ = info
        self.n_features_in_ = n_features
        return self


# ---- sparse-group lasso (convex) ----------------------------------------


class SparseGroupLassoRegressor(_GroupSingleLambdaBase):
    """Sparse-group lasso at a single λ."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        lambda_: float = 0.1,
        alpha: float = 0.5,
        *,
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        screening: str = "strong",
        acceleration: int | None = 5,
        parallel: bool = False,
    ) -> None:
        self.groups = groups
        self.lambda_ = lambda_
        self.alpha = alpha
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.screening = screening
        self.acceleration = acceleration
        self.parallel = parallel

    def fit(self, x, y: NDArray[np.float64]) -> "SparseGroupLassoRegressor":
        common, sparse_payload, n_features = _ls_group_dispatch_inputs(
            self, x, y, self.groups, is_path=False,
        )
        common["alpha"] = self.alpha
        if sparse_payload is not None:
            data, indices, indptr, n_rows, n_cols, y_arr, groups_arr = sparse_payload
            coefs, intercepts, _, info = _core.solve_sparse_group_lasso_ls_path_sparse(
                data, indices, indptr, n_rows, n_cols, y_arr, groups_arr, **common,
            )
        else:
            x_arr = common.pop("_x")
            y_arr = common.pop("_y")
            groups_arr = common.pop("_groups")
            coefs, intercepts, _, info = _core.solve_sparse_group_lasso_ls_path(
                x_arr, y_arr, groups_arr, **common,
            )
        self.coef_ = coefs[0]
        self.intercept_ = float(intercepts[0])
        self.info_ = info
        self.n_features_in_ = n_features
        return self


class SparseGroupLassoPathRegressor(_GroupPathBase):
    """Sparse-group lasso along an entire λ-path."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        alpha: float = 0.5,
        *,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        screening: str = "strong",
        acceleration: int | None = 5,
        parallel: bool = False,
    ) -> None:
        self.groups = groups
        self.alpha = alpha
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.screening = screening
        self.acceleration = acceleration
        self.parallel = parallel

    def fit(self, x, y: NDArray[np.float64]) -> "SparseGroupLassoPathRegressor":
        common, sparse_payload, n_features = _ls_group_dispatch_inputs(
            self, x, y, self.groups, is_path=True,
        )
        common["alpha"] = self.alpha
        if sparse_payload is not None:
            data, indices, indptr, n_rows, n_cols, y_arr, groups_arr = sparse_payload
            coefs, intercepts, lambdas_used, info = _core.solve_sparse_group_lasso_ls_path_sparse(
                data, indices, indptr, n_rows, n_cols, y_arr, groups_arr, **common,
            )
        else:
            x_arr = common.pop("_x")
            y_arr = common.pop("_y")
            groups_arr = common.pop("_groups")
            coefs, intercepts, lambdas_used, info = _core.solve_sparse_group_lasso_ls_path(
                x_arr, y_arr, groups_arr, **common,
            )
        self.coefs_ = coefs
        self.intercepts_ = intercepts
        self.lambdas_ = lambdas_used
        self.info_ = info
        self.n_features_in_ = n_features
        return self


# ---- group elastic net (convex) ----------------------------------------


class GroupElasticNetRegressor(_GroupSingleLambdaBase):
    """Group elastic net at a single λ.

    Convex per-group penalty `α λ w_g ‖β_g‖₂ + (1-α) λ w_g ‖β_g‖₂² / 2`.
    `α = 1` reduces to plain group lasso; `α = 0` is per-block ridge.
    """

    def __init__(
        self,
        groups: NDArray[np.int64],
        lambda_: float = 0.1,
        alpha: float = 0.5,
        *,
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        screening: str = "strong",
        acceleration: int | None = 5,
        parallel: bool = False,
    ) -> None:
        self.groups = groups
        self.lambda_ = lambda_
        self.alpha = alpha
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.screening = screening
        self.acceleration = acceleration
        self.parallel = parallel

    def fit(self, x, y: NDArray[np.float64]) -> "GroupElasticNetRegressor":
        common, sparse_payload, n_features = _ls_group_dispatch_inputs(
            self, x, y, self.groups, is_path=False,
        )
        common["alpha"] = self.alpha
        if sparse_payload is not None:
            data, indices, indptr, n_rows, n_cols, y_arr, groups_arr = sparse_payload
            coefs, intercepts, _, info = _core.solve_group_elastic_net_ls_path_sparse(
                data, indices, indptr, n_rows, n_cols, y_arr, groups_arr, **common,
            )
        else:
            x_arr = common.pop("_x")
            y_arr = common.pop("_y")
            groups_arr = common.pop("_groups")
            coefs, intercepts, _, info = _core.solve_group_elastic_net_ls_path(
                x_arr, y_arr, groups_arr, **common,
            )
        self.coef_ = coefs[0]
        self.intercept_ = float(intercepts[0])
        self.info_ = info
        self.n_features_in_ = n_features
        return self


class GroupElasticNetPathRegressor(_GroupPathBase):
    """Group elastic net along an entire λ-path with warm starts."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        alpha: float = 0.5,
        *,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        screening: str = "strong",
        acceleration: int | None = 5,
        parallel: bool = False,
    ) -> None:
        self.groups = groups
        self.alpha = alpha
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.screening = screening
        self.acceleration = acceleration
        self.parallel = parallel

    def fit(self, x, y: NDArray[np.float64]) -> "GroupElasticNetPathRegressor":
        common, sparse_payload, n_features = _ls_group_dispatch_inputs(
            self, x, y, self.groups, is_path=True,
        )
        common["alpha"] = self.alpha
        if sparse_payload is not None:
            data, indices, indptr, n_rows, n_cols, y_arr, groups_arr = sparse_payload
            coefs, intercepts, lambdas_used, info = _core.solve_group_elastic_net_ls_path_sparse(
                data, indices, indptr, n_rows, n_cols, y_arr, groups_arr, **common,
            )
        else:
            x_arr = common.pop("_x")
            y_arr = common.pop("_y")
            groups_arr = common.pop("_groups")
            coefs, intercepts, lambdas_used, info = _core.solve_group_elastic_net_ls_path(
                x_arr, y_arr, groups_arr, **common,
            )
        self.coefs_ = coefs
        self.intercepts_ = intercepts
        self.lambdas_ = lambdas_used
        self.info_ = info
        self.n_features_in_ = n_features
        return self


# ---- sparse-group MCP (LLA-wrapped) -------------------------------------


class SparseGroupMCPRegressor(_GroupSingleLambdaBase):
    """Sparse-group MCP at a single λ via LLA outer loop."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        lambda_: float = 0.1,
        gamma: float = 3.0,
        alpha: float = 0.5,
        *,
        weights: NDArray[np.float64] | None = None,
        coord_weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        screening: str = "strong",
        acceleration: int | None = 5,
        parallel: bool = False,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.groups = groups
        self.lambda_ = lambda_
        self.gamma = gamma
        self.alpha = alpha
        self.weights = weights
        self.coord_weights = coord_weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.screening = screening
        self.acceleration = acceleration
        self.parallel = parallel
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def fit(self, x, y: NDArray[np.float64]) -> "SparseGroupMCPRegressor":
        common, sparse_payload, n_features = _ls_group_dispatch_inputs(
            self, x, y, self.groups, is_path=False,
        )
        common["gamma"] = self.gamma
        common["alpha"] = self.alpha
        common["max_outer"] = self.max_outer
        common["outer_tol"] = self.outer_tol
        cw = (
            np.ascontiguousarray(self.coord_weights, dtype=np.float64)
            if self.coord_weights is not None
            else None
        )
        common["coord_weights"] = cw
        if sparse_payload is not None:
            data, indices, indptr, n_rows, n_cols, y_arr, groups_arr = sparse_payload
            coefs, intercepts, _, info = _core.solve_sparse_group_mcp_ls_path_sparse(
                data, indices, indptr, n_rows, n_cols, y_arr, groups_arr, **common,
            )
        else:
            x_arr = common.pop("_x")
            y_arr = common.pop("_y")
            groups_arr = common.pop("_groups")
            coefs, intercepts, _, info = _core.solve_sparse_group_mcp_ls_path(
                x_arr, y_arr, groups_arr, **common,
            )
        self.coef_ = coefs[0]
        self.intercept_ = float(intercepts[0])
        self.info_ = info
        self.n_features_in_ = n_features
        return self


class SparseGroupSCADRegressor(_GroupSingleLambdaBase):
    """Sparse-group SCAD at a single λ via LLA outer loop. SCAD shape `a > 2`
    (default 3.7 — Fan & Li's recommendation)."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        lambda_: float = 0.1,
        a: float = 3.7,
        alpha: float = 0.5,
        *,
        weights: NDArray[np.float64] | None = None,
        coord_weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        screening: str = "strong",
        acceleration: int | None = 5,
        parallel: bool = False,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.groups = groups
        self.lambda_ = lambda_
        self.a = a
        self.alpha = alpha
        self.weights = weights
        self.coord_weights = coord_weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.screening = screening
        self.acceleration = acceleration
        self.parallel = parallel
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def fit(self, x, y: NDArray[np.float64]) -> "SparseGroupSCADRegressor":
        common, sparse_payload, n_features = _ls_group_dispatch_inputs(
            self, x, y, self.groups, is_path=False,
        )
        common["a"] = self.a
        common["alpha"] = self.alpha
        common["max_outer"] = self.max_outer
        common["outer_tol"] = self.outer_tol
        cw = (
            np.ascontiguousarray(self.coord_weights, dtype=np.float64)
            if self.coord_weights is not None
            else None
        )
        common["coord_weights"] = cw
        if sparse_payload is not None:
            data, indices, indptr, n_rows, n_cols, y_arr, groups_arr = sparse_payload
            coefs, intercepts, _, info = _core.solve_sparse_group_scad_ls_path_sparse(
                data, indices, indptr, n_rows, n_cols, y_arr, groups_arr, **common,
            )
        else:
            x_arr = common.pop("_x")
            y_arr = common.pop("_y")
            groups_arr = common.pop("_groups")
            coefs, intercepts, _, info = _core.solve_sparse_group_scad_ls_path(
                x_arr, y_arr, groups_arr, **common,
            )
        self.coef_ = coefs[0]
        self.intercept_ = float(intercepts[0])
        self.info_ = info
        self.n_features_in_ = n_features
        return self


class SparseGroupSCADPathRegressor(_GroupPathBase):
    """Sparse-group SCAD along an entire λ-path; LLA at every λ.
    SCAD shape `a > 2` (default 3.7)."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        a: float = 3.7,
        alpha: float = 0.5,
        *,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        weights: NDArray[np.float64] | None = None,
        coord_weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        screening: str = "strong",
        acceleration: int | None = 5,
        parallel: bool = False,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.groups = groups
        self.a = a
        self.alpha = alpha
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.weights = weights
        self.coord_weights = coord_weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.screening = screening
        self.acceleration = acceleration
        self.parallel = parallel
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def fit(self, x, y: NDArray[np.float64]) -> "SparseGroupSCADPathRegressor":
        common, sparse_payload, n_features = _ls_group_dispatch_inputs(
            self, x, y, self.groups, is_path=True,
        )
        common["a"] = self.a
        common["alpha"] = self.alpha
        common["max_outer"] = self.max_outer
        common["outer_tol"] = self.outer_tol
        cw = (
            np.ascontiguousarray(self.coord_weights, dtype=np.float64)
            if self.coord_weights is not None
            else None
        )
        common["coord_weights"] = cw
        if sparse_payload is not None:
            data, indices, indptr, n_rows, n_cols, y_arr, groups_arr = sparse_payload
            coefs, intercepts, lambdas_used, info = _core.solve_sparse_group_scad_ls_path_sparse(
                data, indices, indptr, n_rows, n_cols, y_arr, groups_arr, **common,
            )
        else:
            x_arr = common.pop("_x")
            y_arr = common.pop("_y")
            groups_arr = common.pop("_groups")
            coefs, intercepts, lambdas_used, info = _core.solve_sparse_group_scad_ls_path(
                x_arr, y_arr, groups_arr, **common,
            )
        self.coefs_ = coefs
        self.intercepts_ = intercepts
        self.lambdas_ = lambdas_used
        self.info_ = info
        self.n_features_in_ = n_features
        return self


class SparseGroupMCPPathRegressor(_GroupPathBase):
    """Sparse-group MCP along an entire λ-path; LLA at every λ."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        gamma: float = 3.0,
        alpha: float = 0.5,
        *,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        weights: NDArray[np.float64] | None = None,
        coord_weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        screening: str = "strong",
        acceleration: int | None = 5,
        parallel: bool = False,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.groups = groups
        self.gamma = gamma
        self.alpha = alpha
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.weights = weights
        self.coord_weights = coord_weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.screening = screening
        self.acceleration = acceleration
        self.parallel = parallel
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def fit(self, x, y: NDArray[np.float64]) -> "SparseGroupMCPPathRegressor":
        common, sparse_payload, n_features = _ls_group_dispatch_inputs(
            self, x, y, self.groups, is_path=True,
        )
        common["gamma"] = self.gamma
        common["alpha"] = self.alpha
        common["max_outer"] = self.max_outer
        common["outer_tol"] = self.outer_tol
        cw = (
            np.ascontiguousarray(self.coord_weights, dtype=np.float64)
            if self.coord_weights is not None
            else None
        )
        common["coord_weights"] = cw
        if sparse_payload is not None:
            data, indices, indptr, n_rows, n_cols, y_arr, groups_arr = sparse_payload
            coefs, intercepts, lambdas_used, info = _core.solve_sparse_group_mcp_ls_path_sparse(
                data, indices, indptr, n_rows, n_cols, y_arr, groups_arr, **common,
            )
        else:
            x_arr = common.pop("_x")
            y_arr = common.pop("_y")
            groups_arr = common.pop("_groups")
            coefs, intercepts, lambdas_used, info = _core.solve_sparse_group_mcp_ls_path(
                x_arr, y_arr, groups_arr, **common,
            )
        self.coefs_ = coefs
        self.intercepts_ = intercepts
        self.lambdas_ = lambdas_used
        self.info_ = info
        self.n_features_in_ = n_features
        return self


# =====================================================================
# Logistic regression (binomial logit) via prox-Newton
# =====================================================================


def _sigmoid(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Numerically stable sigmoid; works elementwise."""
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


class _LogisticRegressorBase(BaseEstimator):
    """Single-λ logistic estimator; subclasses pick the penalty."""

    coef_: NDArray[np.float64]
    intercept_: float
    info_: dict[str, Any]
    n_features_in_: int

    def _validate_xy_logistic(
        self, x: NDArray[np.float64], y: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        x = np.ascontiguousarray(x, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"x must be 2D, got shape {x.shape}")
        if y.ndim != 1 or y.shape[0] != x.shape[0]:
            raise ValueError(
                f"y must be 1D with length {x.shape[0]}, got shape {y.shape}"
            )
        if not np.all((y == 0.0) | (y == 1.0)):
            raise ValueError("logistic regression requires y ∈ {0, 1}")
        return x, y

    def decision_function(self, x) -> NDArray[np.float64]:
        """Linear scores η = Xβ + α."""
        if _is_sparse(x):
            return np.asarray(x @ self.coef_).ravel() + self.intercept_
        x = np.ascontiguousarray(x, dtype=np.float64)
        return x @ self.coef_ + self.intercept_

    def predict_proba(self, x) -> NDArray[np.float64]:
        """P(y=1 | x). Returns shape (n_samples,). For sklearn-style
        2-column output (n_samples, 2) compute `np.column_stack([1-p, p])`."""
        return _sigmoid(self.decision_function(x))

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Class labels in {0, 1} thresholded at 0.5."""
        return (self.predict_proba(x) >= 0.5).astype(np.float64)


class _LogisticPathRegressorBase(BaseEstimator):
    """λ-path logistic estimator; subclasses pick the penalty."""

    coefs_: NDArray[np.float64]       # (n_lambdas, p)
    intercepts_: NDArray[np.float64]  # (n_lambdas,)
    lambdas_: NDArray[np.float64]
    info_: dict[str, Any]
    n_features_in_: int

    def _validate_xy_logistic(
        self, x: NDArray[np.float64], y: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        x = np.ascontiguousarray(x, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        if not np.all((y == 0.0) | (y == 1.0)):
            raise ValueError("logistic regression requires y ∈ {0, 1}")
        return x, y

    def decision_function(self, x) -> NDArray[np.float64]:
        """Linear scores per λ: shape (n_samples, n_lambdas)."""
        if _is_sparse(x):
            return np.asarray(x @ self.coefs_.T) + self.intercepts_[None, :]
        x = np.ascontiguousarray(x, dtype=np.float64)
        return x @ self.coefs_.T + self.intercepts_[None, :]

    def predict_proba(self, x) -> NDArray[np.float64]:
        """P(y=1) per λ: shape (n_samples, n_lambdas)."""
        return _sigmoid(self.decision_function(x))

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Class labels per λ: shape (n_samples, n_lambdas)."""
        return (self.predict_proba(x) >= 0.5).astype(np.float64)


class LogisticMCPRegressor(_LogisticRegressorBase):
    """Logistic regression with MCP penalty at a single λ (prox-Newton)."""

    def __init__(
        self,
        lambda_: float = 0.1,
        gamma: float = 3.0,
        *,
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.lambda_ = lambda_
        self.gamma = gamma
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def fit(self, x, y) -> "LogisticMCPRegressor":
        common, payload, n_features = _glm_dispatch_inputs(
            self, x, y, validate_y=_validate_y_binary, is_path=False,
        )
        common["gamma"] = self.gamma
        if payload is not None:
            coefs, intercepts, _, info = _core.solve_logistic_mcp_path_sparse(
                *payload, **common
            )
        else:
            x_arr = common.pop("_x"); y_arr = common.pop("_y")
            coefs, intercepts, _, info = _core.solve_logistic_mcp_path(
                x_arr, y_arr, **common
            )
        self.coef_ = coefs[0]
        self.intercept_ = float(intercepts[0])
        self.info_ = info
        self.n_features_in_ = n_features
        return self


class LogisticMCPPathRegressor(_LogisticPathRegressorBase):
    """Logistic regression with MCP penalty along an entire λ-path."""

    def __init__(
        self,
        gamma: float = 3.0,
        *,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        weights: NDArray[np.float64] | None = None,
        sample_weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.gamma = gamma
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.weights = weights
        self.sample_weights = sample_weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def fit(self, x, y) -> "LogisticMCPPathRegressor":
        _reject_sample_weights_on_non_dense(
            self.sample_weights, x, estimator_kind="LogisticMCPPathRegressor"
        )
        if _is_chunked(x):
            y_arr = np.ascontiguousarray(y, dtype=np.float64)
            if y_arr.ndim != 1 or y_arr.shape[0] != x.n_rows:
                raise ValueError(
                    f"y must be 1D with length {x.n_rows}, got shape {y_arr.shape}"
                )
            _validate_y_binary(y_arr)
            w = (
                np.ascontiguousarray(self.weights, dtype=np.float64)
                if self.weights is not None
                else None
            )
            lams = (
                np.ascontiguousarray(self.lambdas, dtype=np.float64)
                if self.lambdas is not None
                else None
            )
            chunked_entry = (
                _core.solve_logistic_mcp_path_chunked_f32
                if x.dtype == "f32"
                else _core.solve_logistic_mcp_path_chunked
            )
            coefs, intercepts, lambdas_used, info = chunked_entry(
                x.chunks, x.n_cols, y_arr,
                gamma=self.gamma, lambdas=lams,
                n_lambdas=self.n_lambdas, lambda_min_ratio=self.lambda_min_ratio,
                weights=w, max_iter=self.max_iter, tol=self.tol,
                acceleration=self.acceleration,
                fit_intercept=self.fit_intercept,
                standardize_x=self.standardize,
                max_outer=self.max_outer, outer_tol=self.outer_tol,
            )
            self.coefs_ = coefs
            self.intercepts_ = intercepts
            self.lambdas_ = lambdas_used
            self.info_ = info
            self.n_features_in_ = x.n_cols
            return self
        if _is_mmap(x):
            y_arr = np.ascontiguousarray(y, dtype=np.float64)
            if y_arr.ndim != 1 or y_arr.shape[0] != x.n_rows:
                raise ValueError(
                    f"y must be 1D with length {x.n_rows}, got shape {y_arr.shape}"
                )
            _validate_y_binary(y_arr)
            w = (
                np.ascontiguousarray(self.weights, dtype=np.float64)
                if self.weights is not None
                else None
            )
            lams = (
                np.ascontiguousarray(self.lambdas, dtype=np.float64)
                if self.lambdas is not None
                else None
            )
            mmap_entry = (
                _core.solve_logistic_mcp_path_mmap_f32
                if x.dtype == "f32"
                else _core.solve_logistic_mcp_path_mmap
            )
            coefs, intercepts, lambdas_used, info = mmap_entry(
                x.path, x.n_rows, x.n_cols, y_arr,
                gamma=self.gamma, lambdas=lams,
                n_lambdas=self.n_lambdas, lambda_min_ratio=self.lambda_min_ratio,
                weights=w, max_iter=self.max_iter, tol=self.tol,
                acceleration=self.acceleration,
                fit_intercept=self.fit_intercept,
                standardize_x=self.standardize,
                max_outer=self.max_outer, outer_tol=self.outer_tol,
            )
            self.coefs_ = coefs
            self.intercepts_ = intercepts
            self.lambdas_ = lambdas_used
            self.info_ = info
            self.n_features_in_ = x.n_cols
            return self
        common, payload, n_features = _glm_dispatch_inputs(
            self, x, y, validate_y=_validate_y_binary, is_path=True,
        )
        common["gamma"] = self.gamma
        if payload is not None:
            coefs, intercepts, lambdas_used, info = _core.solve_logistic_mcp_path_sparse(
                *payload, **common
            )
        else:
            x_arr = common.pop("_x"); y_arr = common.pop("_y")
            coefs, intercepts, lambdas_used, info = _core.solve_logistic_mcp_path(
                x_arr, y_arr, **common
            )
        self.coefs_ = coefs
        self.intercepts_ = intercepts
        self.lambdas_ = lambdas_used
        self.info_ = info
        self.n_features_in_ = n_features
        return self


class LogisticSCADRegressor(_LogisticRegressorBase):
    """Logistic regression with SCAD penalty at a single λ."""

    def __init__(
        self,
        lambda_: float = 0.1,
        a: float = 3.7,
        *,
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.lambda_ = lambda_
        self.a = a
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def fit(self, x, y) -> "LogisticSCADRegressor":
        common, payload, n_features = _glm_dispatch_inputs(
            self, x, y, validate_y=_validate_y_binary, is_path=False,
        )
        common["a"] = self.a
        if payload is not None:
            coefs, intercepts, _, info = _core.solve_logistic_scad_path_sparse(
                *payload, **common
            )
        else:
            x_arr = common.pop("_x"); y_arr = common.pop("_y")
            coefs, intercepts, _, info = _core.solve_logistic_scad_path(
                x_arr, y_arr, **common
            )
        self.coef_ = coefs[0]
        self.intercept_ = float(intercepts[0])
        self.info_ = info
        self.n_features_in_ = n_features
        return self


class LogisticSCADPathRegressor(_LogisticPathRegressorBase):
    """Logistic regression with SCAD penalty along an entire λ-path."""

    def __init__(
        self,
        a: float = 3.7,
        *,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        weights: NDArray[np.float64] | None = None,
        sample_weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.a = a
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.weights = weights
        self.sample_weights = sample_weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def fit(self, x, y) -> "LogisticSCADPathRegressor":
        common, payload, n_features = _glm_dispatch_inputs(
            self, x, y, validate_y=_validate_y_binary, is_path=True,
        )
        common["a"] = self.a
        if payload is not None:
            coefs, intercepts, lambdas_used, info = _core.solve_logistic_scad_path_sparse(
                *payload, **common
            )
        else:
            x_arr = common.pop("_x"); y_arr = common.pop("_y")
            coefs, intercepts, lambdas_used, info = _core.solve_logistic_scad_path(
                x_arr, y_arr, **common
            )
        self.coefs_ = coefs
        self.intercepts_ = intercepts
        self.lambdas_ = lambdas_used
        self.info_ = info
        self.n_features_in_ = n_features
        return self


class LogisticElasticNetRegressor(_LogisticRegressorBase):
    """Logistic regression with elastic-net penalty at a single λ.

    Convex penalty ``α λ |β_j| + (1 - α) λ β_j² / 2`` per feature.
    ``alpha = 1`` is pure logistic lasso; ``alpha = 0`` is ridge. The
    `LogisticLassoRegressor` subclass pins ``alpha = 1`` for convenience.

    Solver: prox-Newton with the existing weighted-LS surrogate
    (`build_glm_path_outputs`) — same machinery as the MCP / SCAD
    siblings, but with a convex `ElasticNet` penalty so the path is
    one-shot rather than the LLA outer loop. ``max_outer`` / ``outer_tol``
    still control the prox-Newton iteration count.
    """

    def __init__(
        self,
        lambda_: float = 0.1,
        alpha: float = 0.5,
        *,
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.lambda_ = lambda_
        self.alpha = alpha
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def fit(self, x, y) -> "LogisticElasticNetRegressor":
        common, payload, n_features = _glm_dispatch_inputs(
            self, x, y, validate_y=_validate_y_binary, is_path=False,
        )
        common["alpha"] = self.alpha
        if payload is not None:
            coefs, intercepts, _, info = _core.solve_logistic_elastic_net_path_sparse(
                *payload, **common
            )
        else:
            x_arr = common.pop("_x"); y_arr = common.pop("_y")
            coefs, intercepts, _, info = _core.solve_logistic_elastic_net_path(
                x_arr, y_arr, **common
            )
        self.coef_ = coefs[0]
        self.intercept_ = float(intercepts[0])
        self.info_ = info
        self.n_features_in_ = n_features
        return self


class LogisticElasticNetPathRegressor(_LogisticPathRegressorBase):
    """Logistic regression with elastic-net penalty along an entire λ-path.

    See :class:`LogisticElasticNetRegressor` for the single-λ variant.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        *,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        weights: NDArray[np.float64] | None = None,
        sample_weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.alpha = alpha
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.weights = weights
        self.sample_weights = sample_weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def fit(self, x, y) -> "LogisticElasticNetPathRegressor":
        common, payload, n_features = _glm_dispatch_inputs(
            self, x, y, validate_y=_validate_y_binary, is_path=True,
        )
        common["alpha"] = self.alpha
        if payload is not None:
            coefs, intercepts, lambdas_used, info = (
                _core.solve_logistic_elastic_net_path_sparse(*payload, **common)
            )
        else:
            x_arr = common.pop("_x"); y_arr = common.pop("_y")
            coefs, intercepts, lambdas_used, info = (
                _core.solve_logistic_elastic_net_path(x_arr, y_arr, **common)
            )
        self.coefs_ = coefs
        self.intercepts_ = intercepts
        self.lambdas_ = lambdas_used
        self.info_ = info
        self.n_features_in_ = n_features
        return self


class LogisticLassoRegressor(LogisticElasticNetRegressor):
    """Logistic regression with L1 penalty at a single λ.

    Convex L1 — proper logistic lasso via prox-Newton with the
    `ElasticNet` penalty pinned to ``alpha = 1``. Use this rather than
    `LogisticMCPRegressor` at very large gamma; the latter pays for
    nonconvex LLA machinery to approximate this convex problem.
    """

    def __init__(
        self,
        lambda_: float = 0.1,
        *,
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        super().__init__(
            lambda_=lambda_, alpha=1.0, weights=weights,
            max_iter=max_iter, tol=tol, fit_intercept=fit_intercept,
            standardize=standardize, acceleration=acceleration,
            max_outer=max_outer, outer_tol=outer_tol,
        )


class LogisticLassoPathRegressor(LogisticElasticNetPathRegressor):
    """Logistic regression with L1 penalty along an entire λ-path.

    See :class:`LogisticLassoRegressor` for the single-λ variant.
    """

    def __init__(
        self,
        *,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        weights: NDArray[np.float64] | None = None,
        sample_weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        super().__init__(
            alpha=1.0, lambdas=lambdas, n_lambdas=n_lambdas,
            lambda_min_ratio=lambda_min_ratio, weights=weights,
            sample_weights=sample_weights, max_iter=max_iter, tol=tol,
            fit_intercept=fit_intercept, standardize=standardize,
            acceleration=acceleration, max_outer=max_outer, outer_tol=outer_tol,
        )


# =====================================================================
# Logistic + group penalties (M3.3)
# =====================================================================


class _LogisticGroupSingleLambdaBase(_LogisticRegressorBase, _GroupEstimatorMixin):
    """Common base for single-λ logistic+group regressors."""

    pass


class _LogisticGroupPathBase(_LogisticPathRegressorBase, _GroupEstimatorMixin):
    """Common base for full-path logistic+group regressors."""

    pass


# ---- logistic + group lasso ---------------------------------------------


class LogisticGroupLassoRegressor(_LogisticGroupSingleLambdaBase):
    """Logistic regression with group lasso penalty at a single λ."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        lambda_: float = 0.1,
        *,
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.groups = groups
        self.lambda_ = lambda_
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def fit(self, x, y) -> "LogisticGroupLassoRegressor":
        common, payload, n_features = _glm_dispatch_inputs(
            self, x, y, validate_y=_validate_y_binary, is_path=False, groups=self.groups,
        )
        if payload is not None:
            coefs, intercepts, _, info = _core.solve_logistic_group_lasso_path_sparse(
                *payload, **common
            )
        else:
            x_arr = common.pop("_x"); y_arr = common.pop("_y"); g_arr = common.pop("_groups")
            coefs, intercepts, _, info = _core.solve_logistic_group_lasso_path(
                x_arr, y_arr, g_arr, **common
            )
        self.coef_ = coefs[0]
        self.intercept_ = float(intercepts[0])
        self.info_ = info
        self.n_features_in_ = n_features
        return self


class LogisticGroupLassoPathRegressor(_LogisticGroupPathBase):
    """Logistic regression with group lasso along an entire λ-path."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        *,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.groups = groups
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def fit(self, x, y) -> "LogisticGroupLassoPathRegressor":
        common, payload, n_features = _glm_dispatch_inputs(
            self, x, y, validate_y=_validate_y_binary, is_path=True, groups=self.groups,
        )
        if payload is not None:
            coefs, intercepts, lambdas_used, info = _core.solve_logistic_group_lasso_path_sparse(
                *payload, **common
            )
        else:
            x_arr = common.pop("_x"); y_arr = common.pop("_y"); g_arr = common.pop("_groups")
            coefs, intercepts, lambdas_used, info = _core.solve_logistic_group_lasso_path(
                x_arr, y_arr, g_arr, **common
            )
        self.coefs_ = coefs
        self.intercepts_ = intercepts
        self.lambdas_ = lambdas_used
        self.info_ = info
        self.n_features_in_ = n_features
        return self


# ---- logistic + group MCP (LLA) -----------------------------------------


class LogisticGroupMCPRegressor(_LogisticGroupSingleLambdaBase):
    """Logistic regression with group MCP at a single λ (prox-Newton + LLA)."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        lambda_: float = 0.1,
        gamma: float = 3.0,
        *,
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.groups = groups
        self.lambda_ = lambda_
        self.gamma = gamma
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def fit(self, x, y) -> "LogisticGroupMCPRegressor":
        common, payload, n_features = _glm_dispatch_inputs(
            self, x, y, validate_y=_validate_y_binary, is_path=False, groups=self.groups,
        )
        common["gamma"] = self.gamma
        if payload is not None:
            coefs, intercepts, _, info = _core.solve_logistic_group_mcp_path_sparse(
                *payload, **common
            )
        else:
            x_arr = common.pop("_x"); y_arr = common.pop("_y"); g_arr = common.pop("_groups")
            coefs, intercepts, _, info = _core.solve_logistic_group_mcp_path(
                x_arr, y_arr, g_arr, **common
            )
        self.coef_ = coefs[0]
        self.intercept_ = float(intercepts[0])
        self.info_ = info
        self.n_features_in_ = n_features
        return self


class LogisticGroupMCPPathRegressor(_LogisticGroupPathBase):
    """Logistic regression with group MCP along an entire λ-path."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        gamma: float = 3.0,
        *,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.groups = groups
        self.gamma = gamma
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def fit(self, x, y) -> "LogisticGroupMCPPathRegressor":
        common, payload, n_features = _glm_dispatch_inputs(
            self, x, y, validate_y=_validate_y_binary, is_path=True, groups=self.groups,
        )
        common["gamma"] = self.gamma
        if payload is not None:
            coefs, intercepts, lambdas_used, info = _core.solve_logistic_group_mcp_path_sparse(
                *payload, **common
            )
        else:
            x_arr = common.pop("_x"); y_arr = common.pop("_y"); g_arr = common.pop("_groups")
            coefs, intercepts, lambdas_used, info = _core.solve_logistic_group_mcp_path(
                x_arr, y_arr, g_arr, **common
            )
        self.coefs_ = coefs
        self.intercepts_ = intercepts
        self.lambdas_ = lambdas_used
        self.info_ = info
        self.n_features_in_ = n_features
        return self


# ---- logistic + sparse-group lasso --------------------------------------


class LogisticSparseGroupLassoRegressor(_LogisticGroupSingleLambdaBase):
    """Logistic regression with sparse-group lasso at a single λ."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        lambda_: float = 0.1,
        alpha: float = 0.5,
        *,
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.groups = groups
        self.lambda_ = lambda_
        self.alpha = alpha
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def fit(self, x, y) -> "LogisticSparseGroupLassoRegressor":
        common, payload, n_features = _glm_dispatch_inputs(
            self, x, y, validate_y=_validate_y_binary, is_path=False, groups=self.groups,
        )
        common["alpha"] = self.alpha
        if payload is not None:
            coefs, intercepts, _, info = _core.solve_logistic_sparse_group_lasso_path_sparse(
                *payload, **common
            )
        else:
            x_arr = common.pop("_x"); y_arr = common.pop("_y"); g_arr = common.pop("_groups")
            coefs, intercepts, _, info = _core.solve_logistic_sparse_group_lasso_path(
                x_arr, y_arr, g_arr, **common
            )
        self.coef_ = coefs[0]
        self.intercept_ = float(intercepts[0])
        self.info_ = info
        self.n_features_in_ = n_features
        return self


class LogisticSparseGroupLassoPathRegressor(_LogisticGroupPathBase):
    """Logistic regression with sparse-group lasso along an entire λ-path."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        alpha: float = 0.5,
        *,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.groups = groups
        self.alpha = alpha
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def fit(self, x, y) -> "LogisticSparseGroupLassoPathRegressor":
        common, payload, n_features = _glm_dispatch_inputs(
            self, x, y, validate_y=_validate_y_binary, is_path=True, groups=self.groups,
        )
        common["alpha"] = self.alpha
        if payload is not None:
            coefs, intercepts, lambdas_used, info = _core.solve_logistic_sparse_group_lasso_path_sparse(
                *payload, **common
            )
        else:
            x_arr = common.pop("_x"); y_arr = common.pop("_y"); g_arr = common.pop("_groups")
            coefs, intercepts, lambdas_used, info = _core.solve_logistic_sparse_group_lasso_path(
                x_arr, y_arr, g_arr, **common
            )
        self.coefs_ = coefs
        self.intercepts_ = intercepts
        self.lambdas_ = lambdas_used
        self.info_ = info
        self.n_features_in_ = n_features
        return self


# ---- logistic + sparse-group MCP (LLA) ----------------------------------


class LogisticSparseGroupMCPRegressor(_LogisticGroupSingleLambdaBase):
    """Logistic regression with sparse-group MCP at a single λ."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        lambda_: float = 0.1,
        gamma: float = 3.0,
        alpha: float = 0.5,
        *,
        weights: NDArray[np.float64] | None = None,
        coord_weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.groups = groups
        self.lambda_ = lambda_
        self.gamma = gamma
        self.alpha = alpha
        self.weights = weights
        self.coord_weights = coord_weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def fit(self, x, y) -> "LogisticSparseGroupMCPRegressor":
        common, payload, n_features = _glm_dispatch_inputs(
            self, x, y, validate_y=_validate_y_binary, is_path=False, groups=self.groups,
        )
        common["gamma"] = self.gamma
        common["alpha"] = self.alpha
        common["coord_weights"] = (
            np.ascontiguousarray(self.coord_weights, dtype=np.float64)
            if self.coord_weights is not None
            else None
        )
        if payload is not None:
            coefs, intercepts, _, info = _core.solve_logistic_sparse_group_mcp_path_sparse(
                *payload, **common
            )
        else:
            x_arr = common.pop("_x"); y_arr = common.pop("_y"); g_arr = common.pop("_groups")
            coefs, intercepts, _, info = _core.solve_logistic_sparse_group_mcp_path(
                x_arr, y_arr, g_arr, **common
            )
        self.coef_ = coefs[0]
        self.intercept_ = float(intercepts[0])
        self.info_ = info
        self.n_features_in_ = n_features
        return self


class LogisticSparseGroupMCPPathRegressor(_LogisticGroupPathBase):
    """Logistic regression with sparse-group MCP along an entire λ-path."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        gamma: float = 3.0,
        alpha: float = 0.5,
        *,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        weights: NDArray[np.float64] | None = None,
        coord_weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.groups = groups
        self.gamma = gamma
        self.alpha = alpha
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.weights = weights
        self.coord_weights = coord_weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def fit(self, x, y) -> "LogisticSparseGroupMCPPathRegressor":
        common, payload, n_features = _glm_dispatch_inputs(
            self, x, y, validate_y=_validate_y_binary, is_path=True, groups=self.groups,
        )
        common["gamma"] = self.gamma
        common["alpha"] = self.alpha
        common["coord_weights"] = (
            np.ascontiguousarray(self.coord_weights, dtype=np.float64)
            if self.coord_weights is not None
            else None
        )
        if payload is not None:
            coefs, intercepts, lambdas_used, info = _core.solve_logistic_sparse_group_mcp_path_sparse(
                *payload, **common
            )
        else:
            x_arr = common.pop("_x"); y_arr = common.pop("_y"); g_arr = common.pop("_groups")
            coefs, intercepts, lambdas_used, info = _core.solve_logistic_sparse_group_mcp_path(
                x_arr, y_arr, g_arr, **common
            )
        self.coefs_ = coefs
        self.intercepts_ = intercepts
        self.lambdas_ = lambdas_used
        self.info_ = info
        self.n_features_in_ = n_features
        return self


class LogisticSparseGroupSCADRegressor(_LogisticGroupSingleLambdaBase):
    """Logistic regression with sparse-group SCAD at a single λ.
    SCAD shape `a > 2` (default 3.7)."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        lambda_: float = 0.1,
        a: float = 3.7,
        alpha: float = 0.5,
        *,
        weights: NDArray[np.float64] | None = None,
        coord_weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.groups = groups
        self.lambda_ = lambda_
        self.a = a
        self.alpha = alpha
        self.weights = weights
        self.coord_weights = coord_weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def fit(self, x, y) -> "LogisticSparseGroupSCADRegressor":
        common, payload, n_features = _glm_dispatch_inputs(
            self, x, y, validate_y=_validate_y_binary, is_path=False, groups=self.groups,
        )
        common["a"] = self.a
        common["alpha"] = self.alpha
        common["coord_weights"] = (
            np.ascontiguousarray(self.coord_weights, dtype=np.float64)
            if self.coord_weights is not None
            else None
        )
        if payload is not None:
            coefs, intercepts, _, info = _core.solve_logistic_sparse_group_scad_path_sparse(
                *payload, **common
            )
        else:
            x_arr = common.pop("_x"); y_arr = common.pop("_y"); g_arr = common.pop("_groups")
            coefs, intercepts, _, info = _core.solve_logistic_sparse_group_scad_path(
                x_arr, y_arr, g_arr, **common
            )
        self.coef_ = coefs[0]
        self.intercept_ = float(intercepts[0])
        self.info_ = info
        self.n_features_in_ = n_features
        return self


class LogisticSparseGroupSCADPathRegressor(_LogisticGroupPathBase):
    """Logistic regression with sparse-group SCAD along an entire λ-path."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        a: float = 3.7,
        alpha: float = 0.5,
        *,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        weights: NDArray[np.float64] | None = None,
        coord_weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.groups = groups
        self.a = a
        self.alpha = alpha
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.weights = weights
        self.coord_weights = coord_weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def fit(self, x, y) -> "LogisticSparseGroupSCADPathRegressor":
        common, payload, n_features = _glm_dispatch_inputs(
            self, x, y, validate_y=_validate_y_binary, is_path=True, groups=self.groups,
        )
        common["a"] = self.a
        common["alpha"] = self.alpha
        common["coord_weights"] = (
            np.ascontiguousarray(self.coord_weights, dtype=np.float64)
            if self.coord_weights is not None
            else None
        )
        if payload is not None:
            coefs, intercepts, lambdas_used, info = _core.solve_logistic_sparse_group_scad_path_sparse(
                *payload, **common
            )
        else:
            x_arr = common.pop("_x"); y_arr = common.pop("_y"); g_arr = common.pop("_groups")
            coefs, intercepts, lambdas_used, info = _core.solve_logistic_sparse_group_scad_path(
                x_arr, y_arr, g_arr, **common
            )
        self.coefs_ = coefs
        self.intercepts_ = intercepts
        self.lambdas_ = lambdas_used
        self.info_ = info
        self.n_features_in_ = n_features
        return self


# =====================================================================
# Poisson regression (log link) via prox-Newton (M3.4)
# =====================================================================


class _PoissonRegressorBase(BaseEstimator, RegressorMixin):
    """Single-λ Poisson estimator; subclasses pick the penalty.

    `predict(x)` returns the conditional mean μ = exp(η), matching
    sklearn's `PoissonRegressor` convention. `decision_function(x)`
    returns the linear predictor η = Xβ + α (log-rate)."""

    coef_: NDArray[np.float64]
    intercept_: float
    info_: dict[str, Any]
    n_features_in_: int

    def _validate_xy_poisson(
        self, x: NDArray[np.float64], y: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        x = np.ascontiguousarray(x, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"x must be 2D, got shape {x.shape}")
        if y.ndim != 1 or y.shape[0] != x.shape[0]:
            raise ValueError(
                f"y must be 1D with length {x.shape[0]}, got shape {y.shape}"
            )
        if not np.all(np.isfinite(y)) or np.any(y < 0.0):
            raise ValueError("Poisson regression requires y ≥ 0 (finite)")
        return x, y

    def decision_function(self, x) -> NDArray[np.float64]:
        """Linear predictor η = Xβ + α (log-rate)."""
        if _is_sparse(x):
            return np.asarray(x @ self.coef_).ravel() + self.intercept_
        x = np.ascontiguousarray(x, dtype=np.float64)
        return x @ self.coef_ + self.intercept_

    def predict(self, x) -> NDArray[np.float64]:
        """Conditional mean μ = exp(η) (the predicted rate / count)."""
        return np.exp(self.decision_function(x))


class _PoissonPathRegressorBase(BaseEstimator):
    """λ-path Poisson estimator; subclasses pick the penalty."""

    coefs_: NDArray[np.float64]       # (n_lambdas, p)
    intercepts_: NDArray[np.float64]  # (n_lambdas,)
    lambdas_: NDArray[np.float64]
    info_: dict[str, Any]
    n_features_in_: int

    def _validate_xy_poisson(
        self, x: NDArray[np.float64], y: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        x = np.ascontiguousarray(x, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        if not np.all(np.isfinite(y)) or np.any(y < 0.0):
            raise ValueError("Poisson regression requires y ≥ 0 (finite)")
        return x, y

    def decision_function(self, x) -> NDArray[np.float64]:
        """Linear predictor per λ: shape (n_samples, n_lambdas)."""
        if _is_sparse(x):
            return np.asarray(x @ self.coefs_.T) + self.intercepts_[None, :]
        x = np.ascontiguousarray(x, dtype=np.float64)
        return x @ self.coefs_.T + self.intercepts_[None, :]

    def predict(self, x) -> NDArray[np.float64]:
        """Predicted rate per λ: shape (n_samples, n_lambdas)."""
        return np.exp(self.decision_function(x))


class PoissonMCPRegressor(_PoissonRegressorBase):
    """Poisson regression with MCP penalty at a single λ (prox-Newton)."""

    def __init__(
        self,
        lambda_: float = 0.1,
        gamma: float = 3.0,
        *,
        offset: NDArray[np.float64] | None = None,
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.lambda_ = lambda_
        self.gamma = gamma
        self.offset = offset
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def fit(self, x, y) -> "PoissonMCPRegressor":
        common, payload, n_features = _glm_dispatch_inputs(
            self, x, y, validate_y=_validate_y_nonneg, is_path=False,
        )
        common["gamma"] = self.gamma
        if payload is not None:
            coefs, intercepts, _, info = _core.solve_poisson_mcp_path_sparse(
                *payload, **common
            )
        else:
            x_arr = common.pop("_x"); y_arr = common.pop("_y")
            coefs, intercepts, _, info = _core.solve_poisson_mcp_path(
                x_arr, y_arr, **common
            )
        self.coef_ = coefs[0]
        self.intercept_ = float(intercepts[0])
        self.info_ = info
        self.n_features_in_ = n_features
        return self


class PoissonMCPPathRegressor(_PoissonPathRegressorBase):
    """Poisson regression with MCP penalty along an entire λ-path."""

    def __init__(
        self,
        gamma: float = 3.0,
        *,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        offset: NDArray[np.float64] | None = None,
        weights: NDArray[np.float64] | None = None,
        sample_weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.gamma = gamma
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.offset = offset
        self.weights = weights
        self.sample_weights = sample_weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def fit(self, x, y) -> "PoissonMCPPathRegressor":
        common, payload, n_features = _glm_dispatch_inputs(
            self, x, y, validate_y=_validate_y_nonneg, is_path=True,
        )
        common["gamma"] = self.gamma
        if payload is not None:
            coefs, intercepts, lambdas_used, info = _core.solve_poisson_mcp_path_sparse(
                *payload, **common
            )
        else:
            x_arr = common.pop("_x"); y_arr = common.pop("_y")
            coefs, intercepts, lambdas_used, info = _core.solve_poisson_mcp_path(
                x_arr, y_arr, **common
            )
        self.coefs_ = coefs
        self.intercepts_ = intercepts
        self.lambdas_ = lambdas_used
        self.info_ = info
        self.n_features_in_ = n_features
        return self


class PoissonSCADRegressor(_PoissonRegressorBase):
    """Poisson regression with SCAD penalty at a single λ."""

    def __init__(
        self,
        lambda_: float = 0.1,
        a: float = 3.7,
        *,
        offset: NDArray[np.float64] | None = None,
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.lambda_ = lambda_
        self.a = a
        self.offset = offset
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def fit(self, x, y) -> "PoissonSCADRegressor":
        common, payload, n_features = _glm_dispatch_inputs(
            self, x, y, validate_y=_validate_y_nonneg, is_path=False,
        )
        common["a"] = self.a
        if payload is not None:
            coefs, intercepts, _, info = _core.solve_poisson_scad_path_sparse(
                *payload, **common
            )
        else:
            x_arr = common.pop("_x"); y_arr = common.pop("_y")
            coefs, intercepts, _, info = _core.solve_poisson_scad_path(
                x_arr, y_arr, **common
            )
        self.coef_ = coefs[0]
        self.intercept_ = float(intercepts[0])
        self.info_ = info
        self.n_features_in_ = n_features
        return self


class PoissonSCADPathRegressor(_PoissonPathRegressorBase):
    """Poisson regression with SCAD penalty along an entire λ-path."""

    def __init__(
        self,
        a: float = 3.7,
        *,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        offset: NDArray[np.float64] | None = None,
        weights: NDArray[np.float64] | None = None,
        sample_weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.a = a
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.offset = offset
        self.weights = weights
        self.sample_weights = sample_weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def fit(self, x, y) -> "PoissonSCADPathRegressor":
        common, payload, n_features = _glm_dispatch_inputs(
            self, x, y, validate_y=_validate_y_nonneg, is_path=True,
        )
        common["a"] = self.a
        if payload is not None:
            coefs, intercepts, lambdas_used, info = _core.solve_poisson_scad_path_sparse(
                *payload, **common
            )
        else:
            x_arr = common.pop("_x"); y_arr = common.pop("_y")
            coefs, intercepts, lambdas_used, info = _core.solve_poisson_scad_path(
                x_arr, y_arr, **common
            )
        self.coefs_ = coefs
        self.intercepts_ = intercepts
        self.lambdas_ = lambdas_used
        self.info_ = info
        self.n_features_in_ = n_features
        return self


class PoissonElasticNetRegressor(_PoissonRegressorBase):
    """Poisson regression with elastic-net penalty at a single λ.

    Convex penalty ``α λ |β_j| + (1 - α) λ β_j² / 2`` per feature, with
    log-link Poisson likelihood. ``alpha = 1`` is pure Poisson lasso;
    ``alpha = 0`` is ridge.
    """

    def __init__(
        self,
        lambda_: float = 0.1,
        alpha: float = 0.5,
        *,
        offset: NDArray[np.float64] | None = None,
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.lambda_ = lambda_
        self.alpha = alpha
        self.offset = offset
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def fit(self, x, y) -> "PoissonElasticNetRegressor":
        common, payload, n_features = _glm_dispatch_inputs(
            self, x, y, validate_y=_validate_y_nonneg, is_path=False,
        )
        common["alpha"] = self.alpha
        if payload is not None:
            coefs, intercepts, _, info = _core.solve_poisson_elastic_net_path_sparse(
                *payload, **common
            )
        else:
            x_arr = common.pop("_x"); y_arr = common.pop("_y")
            coefs, intercepts, _, info = _core.solve_poisson_elastic_net_path(
                x_arr, y_arr, **common
            )
        self.coef_ = coefs[0]
        self.intercept_ = float(intercepts[0])
        self.info_ = info
        self.n_features_in_ = n_features
        return self


class PoissonElasticNetPathRegressor(_PoissonPathRegressorBase):
    """Poisson regression with elastic-net penalty along an entire λ-path."""

    def __init__(
        self,
        alpha: float = 0.5,
        *,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        offset: NDArray[np.float64] | None = None,
        weights: NDArray[np.float64] | None = None,
        sample_weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.alpha = alpha
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.offset = offset
        self.weights = weights
        self.sample_weights = sample_weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def fit(self, x, y) -> "PoissonElasticNetPathRegressor":
        common, payload, n_features = _glm_dispatch_inputs(
            self, x, y, validate_y=_validate_y_nonneg, is_path=True,
        )
        common["alpha"] = self.alpha
        if payload is not None:
            coefs, intercepts, lambdas_used, info = (
                _core.solve_poisson_elastic_net_path_sparse(*payload, **common)
            )
        else:
            x_arr = common.pop("_x"); y_arr = common.pop("_y")
            coefs, intercepts, lambdas_used, info = (
                _core.solve_poisson_elastic_net_path(x_arr, y_arr, **common)
            )
        self.coefs_ = coefs
        self.intercepts_ = intercepts
        self.lambdas_ = lambdas_used
        self.info_ = info
        self.n_features_in_ = n_features
        return self


class PoissonLassoRegressor(PoissonElasticNetRegressor):
    """Poisson regression with L1 penalty at a single λ (alpha=1 facade)."""

    def __init__(
        self,
        lambda_: float = 0.1,
        *,
        offset: NDArray[np.float64] | None = None,
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        super().__init__(
            lambda_=lambda_, alpha=1.0, offset=offset, weights=weights,
            max_iter=max_iter, tol=tol, fit_intercept=fit_intercept,
            standardize=standardize, acceleration=acceleration,
            max_outer=max_outer, outer_tol=outer_tol,
        )


class PoissonLassoPathRegressor(PoissonElasticNetPathRegressor):
    """Poisson regression with L1 penalty along an entire λ-path."""

    def __init__(
        self,
        *,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        offset: NDArray[np.float64] | None = None,
        weights: NDArray[np.float64] | None = None,
        sample_weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        super().__init__(
            alpha=1.0, lambdas=lambdas, n_lambdas=n_lambdas,
            lambda_min_ratio=lambda_min_ratio, offset=offset, weights=weights,
            sample_weights=sample_weights, max_iter=max_iter, tol=tol,
            fit_intercept=fit_intercept, standardize=standardize,
            acceleration=acceleration, max_outer=max_outer, outer_tol=outer_tol,
        )


class _PoissonGroupSingleLambdaBase(_PoissonRegressorBase, _GroupEstimatorMixin):
    """Common base for single-λ Poisson + group regressors."""

    pass


class _PoissonGroupPathBase(_PoissonPathRegressorBase, _GroupEstimatorMixin):
    """Common base for full-path Poisson + group regressors."""

    pass


class PoissonGroupLassoRegressor(_PoissonGroupSingleLambdaBase):
    """Poisson regression with group lasso penalty at a single λ."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        lambda_: float = 0.1,
        *,
        offset: NDArray[np.float64] | None = None,
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.groups = groups
        self.lambda_ = lambda_
        self.offset = offset
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def fit(self, x, y) -> "PoissonGroupLassoRegressor":
        common, payload, n_features = _glm_dispatch_inputs(
            self, x, y, validate_y=_validate_y_nonneg, is_path=False, groups=self.groups,
        )
        if payload is not None:
            coefs, intercepts, _, info = _core.solve_poisson_group_lasso_path_sparse(
                *payload, **common
            )
        else:
            x_arr = common.pop("_x"); y_arr = common.pop("_y"); g_arr = common.pop("_groups")
            coefs, intercepts, _, info = _core.solve_poisson_group_lasso_path(
                x_arr, y_arr, g_arr, **common
            )
        self.coef_ = coefs[0]
        self.intercept_ = float(intercepts[0])
        self.info_ = info
        self.n_features_in_ = n_features
        return self


class PoissonGroupLassoPathRegressor(_PoissonGroupPathBase):
    """Poisson regression with group lasso along an entire λ-path."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        *,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        offset: NDArray[np.float64] | None = None,
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.groups = groups
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.offset = offset
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def fit(self, x, y) -> "PoissonGroupLassoPathRegressor":
        common, payload, n_features = _glm_dispatch_inputs(
            self, x, y, validate_y=_validate_y_nonneg, is_path=True, groups=self.groups,
        )
        if payload is not None:
            coefs, intercepts, lambdas_used, info = _core.solve_poisson_group_lasso_path_sparse(
                *payload, **common
            )
        else:
            x_arr = common.pop("_x"); y_arr = common.pop("_y"); g_arr = common.pop("_groups")
            coefs, intercepts, lambdas_used, info = _core.solve_poisson_group_lasso_path(
                x_arr, y_arr, g_arr, **common
            )
        self.coefs_ = coefs
        self.intercepts_ = intercepts
        self.lambdas_ = lambdas_used
        self.info_ = info
        self.n_features_in_ = n_features
        return self


class PoissonGroupMCPRegressor(_PoissonGroupSingleLambdaBase):
    """Poisson regression with group MCP at a single λ (prox-Newton + LLA)."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        lambda_: float = 0.1,
        gamma: float = 3.0,
        *,
        offset: NDArray[np.float64] | None = None,
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.groups = groups
        self.lambda_ = lambda_
        self.gamma = gamma
        self.offset = offset
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def fit(self, x, y) -> "PoissonGroupMCPRegressor":
        common, payload, n_features = _glm_dispatch_inputs(
            self, x, y, validate_y=_validate_y_nonneg, is_path=False, groups=self.groups,
        )
        common["gamma"] = self.gamma
        if payload is not None:
            coefs, intercepts, _, info = _core.solve_poisson_group_mcp_path_sparse(
                *payload, **common
            )
        else:
            x_arr = common.pop("_x"); y_arr = common.pop("_y"); g_arr = common.pop("_groups")
            coefs, intercepts, _, info = _core.solve_poisson_group_mcp_path(
                x_arr, y_arr, g_arr, **common
            )
        self.coef_ = coefs[0]
        self.intercept_ = float(intercepts[0])
        self.info_ = info
        self.n_features_in_ = n_features
        return self


class PoissonGroupMCPPathRegressor(_PoissonGroupPathBase):
    """Poisson regression with group MCP along an entire λ-path."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        gamma: float = 3.0,
        *,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        offset: NDArray[np.float64] | None = None,
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.groups = groups
        self.gamma = gamma
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.offset = offset
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def fit(self, x, y) -> "PoissonGroupMCPPathRegressor":
        common, payload, n_features = _glm_dispatch_inputs(
            self, x, y, validate_y=_validate_y_nonneg, is_path=True, groups=self.groups,
        )
        common["gamma"] = self.gamma
        if payload is not None:
            coefs, intercepts, lambdas_used, info = _core.solve_poisson_group_mcp_path_sparse(
                *payload, **common
            )
        else:
            x_arr = common.pop("_x"); y_arr = common.pop("_y"); g_arr = common.pop("_groups")
            coefs, intercepts, lambdas_used, info = _core.solve_poisson_group_mcp_path(
                x_arr, y_arr, g_arr, **common
            )
        self.coefs_ = coefs
        self.intercepts_ = intercepts
        self.lambdas_ = lambdas_used
        self.info_ = info
        self.n_features_in_ = n_features
        return self


class PoissonSparseGroupLassoRegressor(_PoissonGroupSingleLambdaBase):
    """Poisson regression with sparse-group lasso at a single λ."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        lambda_: float = 0.1,
        alpha: float = 0.5,
        *,
        offset: NDArray[np.float64] | None = None,
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.groups = groups
        self.lambda_ = lambda_
        self.alpha = alpha
        self.offset = offset
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def fit(self, x, y) -> "PoissonSparseGroupLassoRegressor":
        common, payload, n_features = _glm_dispatch_inputs(
            self, x, y, validate_y=_validate_y_nonneg, is_path=False, groups=self.groups,
        )
        common["alpha"] = self.alpha
        if payload is not None:
            coefs, intercepts, _, info = _core.solve_poisson_sparse_group_lasso_path_sparse(
                *payload, **common
            )
        else:
            x_arr = common.pop("_x"); y_arr = common.pop("_y"); g_arr = common.pop("_groups")
            coefs, intercepts, _, info = _core.solve_poisson_sparse_group_lasso_path(
                x_arr, y_arr, g_arr, **common
            )
        self.coef_ = coefs[0]
        self.intercept_ = float(intercepts[0])
        self.info_ = info
        self.n_features_in_ = n_features
        return self


class PoissonSparseGroupLassoPathRegressor(_PoissonGroupPathBase):
    """Poisson regression with sparse-group lasso along an entire λ-path."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        alpha: float = 0.5,
        *,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        offset: NDArray[np.float64] | None = None,
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.groups = groups
        self.alpha = alpha
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.offset = offset
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def fit(self, x, y) -> "PoissonSparseGroupLassoPathRegressor":
        common, payload, n_features = _glm_dispatch_inputs(
            self, x, y, validate_y=_validate_y_nonneg, is_path=True, groups=self.groups,
        )
        common["alpha"] = self.alpha
        if payload is not None:
            coefs, intercepts, lambdas_used, info = _core.solve_poisson_sparse_group_lasso_path_sparse(
                *payload, **common
            )
        else:
            x_arr = common.pop("_x"); y_arr = common.pop("_y"); g_arr = common.pop("_groups")
            coefs, intercepts, lambdas_used, info = _core.solve_poisson_sparse_group_lasso_path(
                x_arr, y_arr, g_arr, **common
            )
        self.coefs_ = coefs
        self.intercepts_ = intercepts
        self.lambdas_ = lambdas_used
        self.info_ = info
        self.n_features_in_ = n_features
        return self


class PoissonSparseGroupMCPRegressor(_PoissonGroupSingleLambdaBase):
    """Poisson regression with sparse-group MCP at a single λ."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        lambda_: float = 0.1,
        gamma: float = 3.0,
        alpha: float = 0.5,
        *,
        offset: NDArray[np.float64] | None = None,
        weights: NDArray[np.float64] | None = None,
        coord_weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.groups = groups
        self.lambda_ = lambda_
        self.gamma = gamma
        self.alpha = alpha
        self.offset = offset
        self.weights = weights
        self.coord_weights = coord_weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def fit(self, x, y) -> "PoissonSparseGroupMCPRegressor":
        common, payload, n_features = _glm_dispatch_inputs(
            self, x, y, validate_y=_validate_y_nonneg, is_path=False, groups=self.groups,
        )
        common["gamma"] = self.gamma
        common["alpha"] = self.alpha
        common["coord_weights"] = (
            np.ascontiguousarray(self.coord_weights, dtype=np.float64)
            if self.coord_weights is not None
            else None
        )
        if payload is not None:
            coefs, intercepts, _, info = _core.solve_poisson_sparse_group_mcp_path_sparse(
                *payload, **common
            )
        else:
            x_arr = common.pop("_x"); y_arr = common.pop("_y"); g_arr = common.pop("_groups")
            coefs, intercepts, _, info = _core.solve_poisson_sparse_group_mcp_path(
                x_arr, y_arr, g_arr, **common
            )
        self.coef_ = coefs[0]
        self.intercept_ = float(intercepts[0])
        self.info_ = info
        self.n_features_in_ = n_features
        return self


class PoissonSparseGroupMCPPathRegressor(_PoissonGroupPathBase):
    """Poisson regression with sparse-group MCP along an entire λ-path."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        gamma: float = 3.0,
        alpha: float = 0.5,
        *,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        offset: NDArray[np.float64] | None = None,
        weights: NDArray[np.float64] | None = None,
        coord_weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.groups = groups
        self.gamma = gamma
        self.alpha = alpha
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.offset = offset
        self.weights = weights
        self.coord_weights = coord_weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def fit(self, x, y) -> "PoissonSparseGroupMCPPathRegressor":
        common, payload, n_features = _glm_dispatch_inputs(
            self, x, y, validate_y=_validate_y_nonneg, is_path=True, groups=self.groups,
        )
        common["gamma"] = self.gamma
        common["alpha"] = self.alpha
        common["coord_weights"] = (
            np.ascontiguousarray(self.coord_weights, dtype=np.float64)
            if self.coord_weights is not None
            else None
        )
        if payload is not None:
            coefs, intercepts, lambdas_used, info = _core.solve_poisson_sparse_group_mcp_path_sparse(
                *payload, **common
            )
        else:
            x_arr = common.pop("_x"); y_arr = common.pop("_y"); g_arr = common.pop("_groups")
            coefs, intercepts, lambdas_used, info = _core.solve_poisson_sparse_group_mcp_path(
                x_arr, y_arr, g_arr, **common
            )
        self.coefs_ = coefs
        self.intercepts_ = intercepts
        self.lambdas_ = lambdas_used
        self.info_ = info
        self.n_features_in_ = n_features
        return self


class PoissonSparseGroupSCADRegressor(_PoissonGroupSingleLambdaBase):
    """Poisson regression with sparse-group SCAD at a single λ.
    SCAD shape `a > 2` (default 3.7)."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        lambda_: float = 0.1,
        a: float = 3.7,
        alpha: float = 0.5,
        *,
        offset: NDArray[np.float64] | None = None,
        weights: NDArray[np.float64] | None = None,
        coord_weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.groups = groups
        self.lambda_ = lambda_
        self.a = a
        self.alpha = alpha
        self.offset = offset
        self.weights = weights
        self.coord_weights = coord_weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def fit(self, x, y) -> "PoissonSparseGroupSCADRegressor":
        common, payload, n_features = _glm_dispatch_inputs(
            self, x, y, validate_y=_validate_y_nonneg, is_path=False, groups=self.groups,
        )
        common["a"] = self.a
        common["alpha"] = self.alpha
        common["coord_weights"] = (
            np.ascontiguousarray(self.coord_weights, dtype=np.float64)
            if self.coord_weights is not None
            else None
        )
        if payload is not None:
            coefs, intercepts, _, info = _core.solve_poisson_sparse_group_scad_path_sparse(
                *payload, **common
            )
        else:
            x_arr = common.pop("_x"); y_arr = common.pop("_y"); g_arr = common.pop("_groups")
            coefs, intercepts, _, info = _core.solve_poisson_sparse_group_scad_path(
                x_arr, y_arr, g_arr, **common
            )
        self.coef_ = coefs[0]
        self.intercept_ = float(intercepts[0])
        self.info_ = info
        self.n_features_in_ = n_features
        return self


class PoissonSparseGroupSCADPathRegressor(_PoissonGroupPathBase):
    """Poisson regression with sparse-group SCAD along an entire λ-path."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        a: float = 3.7,
        alpha: float = 0.5,
        *,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        offset: NDArray[np.float64] | None = None,
        weights: NDArray[np.float64] | None = None,
        coord_weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.groups = groups
        self.a = a
        self.alpha = alpha
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.offset = offset
        self.weights = weights
        self.coord_weights = coord_weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def fit(self, x, y) -> "PoissonSparseGroupSCADPathRegressor":
        common, payload, n_features = _glm_dispatch_inputs(
            self, x, y, validate_y=_validate_y_nonneg, is_path=True, groups=self.groups,
        )
        common["a"] = self.a
        common["alpha"] = self.alpha
        common["coord_weights"] = (
            np.ascontiguousarray(self.coord_weights, dtype=np.float64)
            if self.coord_weights is not None
            else None
        )
        if payload is not None:
            coefs, intercepts, lambdas_used, info = _core.solve_poisson_sparse_group_scad_path_sparse(
                *payload, **common
            )
        else:
            x_arr = common.pop("_x"); y_arr = common.pop("_y"); g_arr = common.pop("_groups")
            coefs, intercepts, lambdas_used, info = _core.solve_poisson_sparse_group_scad_path(
                x_arr, y_arr, g_arr, **common
            )
        self.coefs_ = coefs
        self.intercepts_ = intercepts
        self.lambdas_ = lambdas_used
        self.info_ = info
        self.n_features_in_ = n_features
        return self


# =====================================================================
# Cox proportional hazards (Breslow ties) via prox-Newton (M3.5)
# =====================================================================
#
# Cox PH has no per-sample baseline, so estimators don't expose
# `fit_intercept` or `intercept_`. The fit signature is `fit(x, time,
# event)` (3 args) since the outcome is `(time, event)` rather than a
# single `y`. `predict(x) = decision_function(x) = X β` is the prognostic
# index (linear risk score), matching `glmnet::predict.cox`.


class _CoxRegressorBase(BaseEstimator):
    """Single-λ Cox PH estimator; subclasses pick the penalty."""

    coef_: NDArray[np.float64]
    info_: dict[str, Any]
    n_features_in_: int

    def _validate_xte(
        self,
        x: NDArray[np.float64],
        time: NDArray[np.float64],
        event: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        x = np.ascontiguousarray(x, dtype=np.float64)
        time = np.ascontiguousarray(time, dtype=np.float64)
        event = np.ascontiguousarray(event, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"x must be 2D, got shape {x.shape}")
        if time.ndim != 1 or time.shape[0] != x.shape[0]:
            raise ValueError(
                f"time must be 1D with length {x.shape[0]}, got shape {time.shape}"
            )
        if event.shape != time.shape:
            raise ValueError(
                f"event shape {event.shape} must match time shape {time.shape}"
            )
        if not np.all(np.isfinite(time)) or np.any(time < 0.0):
            raise ValueError("Cox PH requires time ≥ 0 (finite)")
        if not np.all((event == 0.0) | (event == 1.0)):
            raise ValueError("Cox PH requires event ∈ {0, 1}")
        if event.sum() < 1:
            raise ValueError("Cox PH requires at least one event (event = 1)")
        return x, time, event

    def decision_function(self, x) -> NDArray[np.float64]:
        """Prognostic index η = Xβ (linear risk score)."""
        if _is_sparse(x):
            return np.asarray(x @ self.coef_).ravel()
        x = np.ascontiguousarray(x, dtype=np.float64)
        return x @ self.coef_

    def predict(self, x) -> NDArray[np.float64]:
        """Same as `decision_function` — Cox has no per-sample baseline,
        so the linear predictor *is* the prognostic index. To convert to
        a relative hazard ratio against a reference, take `exp(predict(x))`."""
        return self.decision_function(x)


class _CoxPathRegressorBase(BaseEstimator):
    """λ-path Cox PH estimator; subclasses pick the penalty."""

    coefs_: NDArray[np.float64]
    lambdas_: NDArray[np.float64]
    info_: dict[str, Any]
    n_features_in_: int

    def _validate_xte(
        self,
        x: NDArray[np.float64],
        time: NDArray[np.float64],
        event: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        return _CoxRegressorBase._validate_xte(self, x, time, event)  # type: ignore[arg-type]

    def decision_function(self, x) -> NDArray[np.float64]:
        """Prognostic index per λ: shape (n_samples, n_lambdas)."""
        if _is_sparse(x):
            return np.asarray(x @ self.coefs_.T)
        x = np.ascontiguousarray(x, dtype=np.float64)
        return x @ self.coefs_.T

    def predict(self, x) -> NDArray[np.float64]:
        return self.decision_function(x)


class CoxMCPRegressor(_CoxRegressorBase):
    """Cox PH regression with MCP penalty at a single λ (prox-Newton)."""

    def __init__(
        self,
        lambda_: float = 0.1,
        gamma: float = 3.0,
        *,
        ties: str = 'breslow',
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        standardize: bool = False,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.lambda_ = lambda_
        self.gamma = gamma
        self.ties = ties
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.standardize = standardize
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def fit(self, x, time, event) -> "CoxMCPRegressor":
        common, payload, n_features = _cox_dispatch_inputs(
            self, x, time, event, is_path=False,
        )
        common["ties"] = self.ties
        common["gamma"] = self.gamma
        if payload is not None:
            coefs, _intercepts, _lambdas, info = _core.solve_cox_mcp_path_sparse(
                *payload, **common
            )
        else:
            x_arr = common.pop("_x"); t_arr = common.pop("_time"); e_arr = common.pop("_event")
            coefs, _intercepts, _lambdas, info = _core.solve_cox_mcp_path(
                x_arr, t_arr, e_arr, **common
            )
        self.coef_ = coefs[0]
        self.info_ = info
        self.n_features_in_ = n_features
        return self


class CoxMCPPathRegressor(_CoxPathRegressorBase):
    """Cox PH regression with MCP penalty along an entire λ-path."""

    def __init__(
        self,
        gamma: float = 3.0,
        *,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        ties: str = 'breslow',
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        standardize: bool = False,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.gamma = gamma
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.ties = ties
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.standardize = standardize
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def fit(self, x, time, event) -> "CoxMCPPathRegressor":
        common, payload, n_features = _cox_dispatch_inputs(
            self, x, time, event, is_path=True,
        )
        common["ties"] = self.ties
        common["gamma"] = self.gamma
        if payload is not None:
            coefs, _intercepts, lambdas_used, info = _core.solve_cox_mcp_path_sparse(
                *payload, **common
            )
        else:
            x_arr = common.pop("_x"); t_arr = common.pop("_time"); e_arr = common.pop("_event")
            coefs, _intercepts, lambdas_used, info = _core.solve_cox_mcp_path(
                x_arr, t_arr, e_arr, **common
            )
        self.coefs_ = coefs
        self.lambdas_ = lambdas_used
        self.info_ = info
        self.n_features_in_ = n_features
        return self


class CoxSCADRegressor(_CoxRegressorBase):
    """Cox PH regression with SCAD penalty at a single λ."""

    def __init__(
        self,
        lambda_: float = 0.1,
        a: float = 3.7,
        *,
        ties: str = 'breslow',
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        standardize: bool = False,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.lambda_ = lambda_
        self.a = a
        self.ties = ties
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.standardize = standardize
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def fit(self, x, time, event) -> "CoxSCADRegressor":
        common, payload, n_features = _cox_dispatch_inputs(
            self, x, time, event, is_path=False,
        )
        common["ties"] = self.ties
        common["a"] = self.a
        if payload is not None:
            coefs, _intercepts, _lambdas, info = _core.solve_cox_scad_path_sparse(
                *payload, **common
            )
        else:
            x_arr = common.pop("_x"); t_arr = common.pop("_time"); e_arr = common.pop("_event")
            coefs, _intercepts, _lambdas, info = _core.solve_cox_scad_path(
                x_arr, t_arr, e_arr, **common
            )
        self.coef_ = coefs[0]
        self.info_ = info
        self.n_features_in_ = n_features
        return self


class CoxSCADPathRegressor(_CoxPathRegressorBase):
    """Cox PH regression with SCAD penalty along an entire λ-path."""

    def __init__(
        self,
        a: float = 3.7,
        *,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        ties: str = 'breslow',
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        standardize: bool = False,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.a = a
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.ties = ties
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.standardize = standardize
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def fit(self, x, time, event) -> "CoxSCADPathRegressor":
        common, payload, n_features = _cox_dispatch_inputs(
            self, x, time, event, is_path=True,
        )
        common["ties"] = self.ties
        common["a"] = self.a
        if payload is not None:
            coefs, _intercepts, lambdas_used, info = _core.solve_cox_scad_path_sparse(
                *payload, **common
            )
        else:
            x_arr = common.pop("_x"); t_arr = common.pop("_time"); e_arr = common.pop("_event")
            coefs, _intercepts, lambdas_used, info = _core.solve_cox_scad_path(
                x_arr, t_arr, e_arr, **common
            )
        self.coefs_ = coefs
        self.lambdas_ = lambdas_used
        self.info_ = info
        self.n_features_in_ = n_features
        return self


class _CoxGroupSingleLambdaBase(_CoxRegressorBase, _GroupEstimatorMixin):
    """Common base for single-λ Cox + group regressors."""

    pass


class _CoxGroupPathBase(_CoxPathRegressorBase, _GroupEstimatorMixin):
    """Common base for full-path Cox + group regressors."""

    pass


class CoxGroupLassoRegressor(_CoxGroupSingleLambdaBase):
    """Cox PH with group lasso at a single λ."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        lambda_: float = 0.1,
        *,
        ties: str = 'breslow',
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        standardize: bool = False,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.groups = groups
        self.lambda_ = lambda_
        self.ties = ties
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.standardize = standardize
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def fit(self, x, time, event) -> "CoxGroupLassoRegressor":
        common, payload, n_features = _cox_dispatch_inputs(
            self, x, time, event, is_path=False, groups=self.groups,
        )
        common["ties"] = self.ties
        if payload is not None:
            coefs, _intercepts, _lambdas, info = _core.solve_cox_group_lasso_path_sparse(
                *payload, **common
            )
        else:
            x_arr = common.pop("_x"); t_arr = common.pop("_time")
            e_arr = common.pop("_event"); g_arr = common.pop("_groups")
            coefs, _intercepts, _lambdas, info = _core.solve_cox_group_lasso_path(
                x_arr, t_arr, e_arr, g_arr, **common
            )
        self.coef_ = coefs[0]
        self.info_ = info
        self.n_features_in_ = n_features
        return self


class CoxGroupLassoPathRegressor(_CoxGroupPathBase):
    """Cox PH with group lasso along an entire λ-path."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        *,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        ties: str = 'breslow',
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        standardize: bool = False,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.groups = groups
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.ties = ties
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.standardize = standardize
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def fit(self, x, time, event) -> "CoxGroupLassoPathRegressor":
        common, payload, n_features = _cox_dispatch_inputs(
            self, x, time, event, is_path=True, groups=self.groups,
        )
        common["ties"] = self.ties
        if payload is not None:
            coefs, _intercepts, lambdas_used, info = _core.solve_cox_group_lasso_path_sparse(
                *payload, **common
            )
        else:
            x_arr = common.pop("_x"); t_arr = common.pop("_time")
            e_arr = common.pop("_event"); g_arr = common.pop("_groups")
            coefs, _intercepts, lambdas_used, info = _core.solve_cox_group_lasso_path(
                x_arr, t_arr, e_arr, g_arr, **common
            )
        self.coefs_ = coefs
        self.lambdas_ = lambdas_used
        self.info_ = info
        self.n_features_in_ = n_features
        return self


class CoxGroupMCPRegressor(_CoxGroupSingleLambdaBase):
    """Cox PH with group MCP at a single λ (prox-Newton + LLA)."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        lambda_: float = 0.1,
        gamma: float = 3.0,
        *,
        ties: str = 'breslow',
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        standardize: bool = False,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.groups = groups
        self.lambda_ = lambda_
        self.gamma = gamma
        self.ties = ties
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.standardize = standardize
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def fit(self, x, time, event) -> "CoxGroupMCPRegressor":
        common, payload, n_features = _cox_dispatch_inputs(
            self, x, time, event, is_path=False, groups=self.groups,
        )
        common["ties"] = self.ties
        common["gamma"] = self.gamma
        if payload is not None:
            coefs, _intercepts, _lambdas, info = _core.solve_cox_group_mcp_path_sparse(
                *payload, **common
            )
        else:
            x_arr = common.pop("_x"); t_arr = common.pop("_time")
            e_arr = common.pop("_event"); g_arr = common.pop("_groups")
            coefs, _intercepts, _lambdas, info = _core.solve_cox_group_mcp_path(
                x_arr, t_arr, e_arr, g_arr, **common
            )
        self.coef_ = coefs[0]
        self.info_ = info
        self.n_features_in_ = n_features
        return self


class CoxGroupMCPPathRegressor(_CoxGroupPathBase):
    """Cox PH with group MCP along an entire λ-path."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        gamma: float = 3.0,
        *,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        ties: str = 'breslow',
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        standardize: bool = False,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.groups = groups
        self.gamma = gamma
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.ties = ties
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.standardize = standardize
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def fit(self, x, time, event) -> "CoxGroupMCPPathRegressor":
        common, payload, n_features = _cox_dispatch_inputs(
            self, x, time, event, is_path=True, groups=self.groups,
        )
        common["ties"] = self.ties
        common["gamma"] = self.gamma
        if payload is not None:
            coefs, _intercepts, lambdas_used, info = _core.solve_cox_group_mcp_path_sparse(
                *payload, **common
            )
        else:
            x_arr = common.pop("_x"); t_arr = common.pop("_time")
            e_arr = common.pop("_event"); g_arr = common.pop("_groups")
            coefs, _intercepts, lambdas_used, info = _core.solve_cox_group_mcp_path(
                x_arr, t_arr, e_arr, g_arr, **common
            )
        self.coefs_ = coefs
        self.lambdas_ = lambdas_used
        self.info_ = info
        self.n_features_in_ = n_features
        return self


class CoxSparseGroupLassoRegressor(_CoxGroupSingleLambdaBase):
    """Cox PH with sparse-group lasso at a single λ."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        lambda_: float = 0.1,
        alpha: float = 0.5,
        *,
        ties: str = 'breslow',
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        standardize: bool = False,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.groups = groups
        self.lambda_ = lambda_
        self.alpha = alpha
        self.ties = ties
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.standardize = standardize
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def fit(self, x, time, event) -> "CoxSparseGroupLassoRegressor":
        common, payload, n_features = _cox_dispatch_inputs(
            self, x, time, event, is_path=False, groups=self.groups,
        )
        common["ties"] = self.ties
        common["alpha"] = self.alpha
        if payload is not None:
            coefs, _intercepts, _lambdas, info = _core.solve_cox_sparse_group_lasso_path_sparse(
                *payload, **common
            )
        else:
            x_arr = common.pop("_x"); t_arr = common.pop("_time")
            e_arr = common.pop("_event"); g_arr = common.pop("_groups")
            coefs, _intercepts, _lambdas, info = _core.solve_cox_sparse_group_lasso_path(
                x_arr, t_arr, e_arr, g_arr, **common
            )
        self.coef_ = coefs[0]
        self.info_ = info
        self.n_features_in_ = n_features
        return self


class CoxSparseGroupLassoPathRegressor(_CoxGroupPathBase):
    """Cox PH with sparse-group lasso along an entire λ-path."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        alpha: float = 0.5,
        *,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        ties: str = 'breslow',
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        standardize: bool = False,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.groups = groups
        self.alpha = alpha
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.ties = ties
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.standardize = standardize
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def fit(self, x, time, event) -> "CoxSparseGroupLassoPathRegressor":
        common, payload, n_features = _cox_dispatch_inputs(
            self, x, time, event, is_path=True, groups=self.groups,
        )
        common["ties"] = self.ties
        common["alpha"] = self.alpha
        if payload is not None:
            coefs, _intercepts, lambdas_used, info = _core.solve_cox_sparse_group_lasso_path_sparse(
                *payload, **common
            )
        else:
            x_arr = common.pop("_x"); t_arr = common.pop("_time")
            e_arr = common.pop("_event"); g_arr = common.pop("_groups")
            coefs, _intercepts, lambdas_used, info = _core.solve_cox_sparse_group_lasso_path(
                x_arr, t_arr, e_arr, g_arr, **common
            )
        self.coefs_ = coefs
        self.lambdas_ = lambdas_used
        self.info_ = info
        self.n_features_in_ = n_features
        return self


class CoxSparseGroupMCPRegressor(_CoxGroupSingleLambdaBase):
    """Cox PH with sparse-group MCP at a single λ."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        lambda_: float = 0.1,
        gamma: float = 3.0,
        alpha: float = 0.5,
        *,
        ties: str = 'breslow',
        weights: NDArray[np.float64] | None = None,
        coord_weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        standardize: bool = False,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.groups = groups
        self.lambda_ = lambda_
        self.gamma = gamma
        self.alpha = alpha
        self.ties = ties
        self.weights = weights
        self.coord_weights = coord_weights
        self.max_iter = max_iter
        self.tol = tol
        self.standardize = standardize
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def fit(self, x, time, event) -> "CoxSparseGroupMCPRegressor":
        common, payload, n_features = _cox_dispatch_inputs(
            self, x, time, event, is_path=False, groups=self.groups,
        )
        common["ties"] = self.ties
        common["gamma"] = self.gamma
        common["alpha"] = self.alpha
        common["coord_weights"] = (
            np.ascontiguousarray(self.coord_weights, dtype=np.float64)
            if self.coord_weights is not None
            else None
        )
        if payload is not None:
            coefs, _intercepts, _lambdas, info = _core.solve_cox_sparse_group_mcp_path_sparse(
                *payload, **common
            )
        else:
            x_arr = common.pop("_x"); t_arr = common.pop("_time")
            e_arr = common.pop("_event"); g_arr = common.pop("_groups")
            coefs, _intercepts, _lambdas, info = _core.solve_cox_sparse_group_mcp_path(
                x_arr, t_arr, e_arr, g_arr, **common
            )
        self.coef_ = coefs[0]
        self.info_ = info
        self.n_features_in_ = n_features
        return self


class CoxSparseGroupMCPPathRegressor(_CoxGroupPathBase):
    """Cox PH with sparse-group MCP along an entire λ-path."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        gamma: float = 3.0,
        alpha: float = 0.5,
        *,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        ties: str = 'breslow',
        weights: NDArray[np.float64] | None = None,
        coord_weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        standardize: bool = False,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.groups = groups
        self.gamma = gamma
        self.alpha = alpha
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.ties = ties
        self.weights = weights
        self.coord_weights = coord_weights
        self.max_iter = max_iter
        self.tol = tol
        self.standardize = standardize
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def fit(self, x, time, event) -> "CoxSparseGroupMCPPathRegressor":
        common, payload, n_features = _cox_dispatch_inputs(
            self, x, time, event, is_path=True, groups=self.groups,
        )
        common["ties"] = self.ties
        common["gamma"] = self.gamma
        common["alpha"] = self.alpha
        common["coord_weights"] = (
            np.ascontiguousarray(self.coord_weights, dtype=np.float64)
            if self.coord_weights is not None
            else None
        )
        if payload is not None:
            coefs, _intercepts, lambdas_used, info = _core.solve_cox_sparse_group_mcp_path_sparse(
                *payload, **common
            )
        else:
            x_arr = common.pop("_x"); t_arr = common.pop("_time")
            e_arr = common.pop("_event"); g_arr = common.pop("_groups")
            coefs, _intercepts, lambdas_used, info = _core.solve_cox_sparse_group_mcp_path(
                x_arr, t_arr, e_arr, g_arr, **common
            )
        self.coefs_ = coefs
        self.lambdas_ = lambdas_used
        self.info_ = info
        self.n_features_in_ = n_features
        return self


class CoxSparseGroupSCADRegressor(_CoxGroupSingleLambdaBase):
    """Cox PH with sparse-group SCAD at a single λ. SCAD shape `a > 2`
    (default 3.7)."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        lambda_: float = 0.1,
        a: float = 3.7,
        alpha: float = 0.5,
        *,
        ties: str = 'breslow',
        weights: NDArray[np.float64] | None = None,
        coord_weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        standardize: bool = False,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.groups = groups
        self.lambda_ = lambda_
        self.a = a
        self.alpha = alpha
        self.ties = ties
        self.weights = weights
        self.coord_weights = coord_weights
        self.max_iter = max_iter
        self.tol = tol
        self.standardize = standardize
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def fit(self, x, time, event) -> "CoxSparseGroupSCADRegressor":
        common, payload, n_features = _cox_dispatch_inputs(
            self, x, time, event, is_path=False, groups=self.groups,
        )
        common["ties"] = self.ties
        common["a"] = self.a
        common["alpha"] = self.alpha
        common["coord_weights"] = (
            np.ascontiguousarray(self.coord_weights, dtype=np.float64)
            if self.coord_weights is not None
            else None
        )
        if payload is not None:
            coefs, _intercepts, _lambdas, info = _core.solve_cox_sparse_group_scad_path_sparse(
                *payload, **common
            )
        else:
            x_arr = common.pop("_x"); t_arr = common.pop("_time")
            e_arr = common.pop("_event"); g_arr = common.pop("_groups")
            coefs, _intercepts, _lambdas, info = _core.solve_cox_sparse_group_scad_path(
                x_arr, t_arr, e_arr, g_arr, **common
            )
        self.coef_ = coefs[0]
        self.info_ = info
        self.n_features_in_ = n_features
        return self


class CoxSparseGroupSCADPathRegressor(_CoxGroupPathBase):
    """Cox PH with sparse-group SCAD along an entire λ-path."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        a: float = 3.7,
        alpha: float = 0.5,
        *,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        ties: str = 'breslow',
        weights: NDArray[np.float64] | None = None,
        coord_weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        standardize: bool = False,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.groups = groups
        self.a = a
        self.alpha = alpha
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.ties = ties
        self.weights = weights
        self.coord_weights = coord_weights
        self.max_iter = max_iter
        self.tol = tol
        self.standardize = standardize
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def fit(self, x, time, event) -> "CoxSparseGroupSCADPathRegressor":
        common, payload, n_features = _cox_dispatch_inputs(
            self, x, time, event, is_path=True, groups=self.groups,
        )
        common["ties"] = self.ties
        common["a"] = self.a
        common["alpha"] = self.alpha
        common["coord_weights"] = (
            np.ascontiguousarray(self.coord_weights, dtype=np.float64)
            if self.coord_weights is not None
            else None
        )
        if payload is not None:
            coefs, _intercepts, lambdas_used, info = _core.solve_cox_sparse_group_scad_path_sparse(
                *payload, **common
            )
        else:
            x_arr = common.pop("_x"); t_arr = common.pop("_time")
            e_arr = common.pop("_event"); g_arr = common.pop("_groups")
            coefs, _intercepts, lambdas_used, info = _core.solve_cox_sparse_group_scad_path(
                x_arr, t_arr, e_arr, g_arr, **common
            )
        self.coefs_ = coefs
        self.lambdas_ = lambdas_used
        self.info_ = info
        self.n_features_in_ = n_features
        return self


# =====================================================================
# Huber regression via prox-Newton (M3.7)
# =====================================================================
#
# Huber is robust LS: the loss is quadratic for residuals within ±δ
# and linear outside, downweighting outliers. Under prox-Newton this
# is iteratively re-weighted least squares with weights `1` if
# `|r_i| ≤ δ` else `δ/|r_i|`, so the M1/M2 inner CD machinery applies
# unchanged via the GlmDatafit trait.
#
# δ is required (no default): users should set it to a robust scale
# estimate of the residual, e.g. 1.345 · σ_MAD(y) for 95% asymptotic
# efficiency at the normal (Huber, 1981). It is a *fixed* hyperparameter
# during a single fit; cross-validating over a few δ values is common.


def _validate_y_finite(y_arr):
    if not np.all(np.isfinite(y_arr)):
        raise ValueError("Huber regression requires finite y (no NaN, no ±∞)")


class _HuberRegressorBase(BaseEstimator, RegressorMixin):
    """Single-λ Huber estimator; subclasses pick the penalty."""

    coef_: NDArray[np.float64]
    intercept_: float
    info_: dict[str, Any]
    n_features_in_: int

    def decision_function(self, x) -> NDArray[np.float64]:
        if _is_sparse(x):
            return np.asarray(x @ self.coef_).ravel() + self.intercept_
        x = np.ascontiguousarray(x, dtype=np.float64)
        return x @ self.coef_ + self.intercept_

    def predict(self, x) -> NDArray[np.float64]:
        return self.decision_function(x)


class _HuberPathRegressorBase(BaseEstimator, RegressorMixin):
    """λ-path Huber estimator; subclasses pick the penalty."""

    coefs_: NDArray[np.float64]       # (n_lambdas, p)
    intercepts_: NDArray[np.float64]  # (n_lambdas,)
    lambdas_: NDArray[np.float64]
    info_: dict[str, Any]
    n_features_in_: int

    def decision_function(self, x) -> NDArray[np.float64]:
        """Linear scores per λ: shape (n_samples, n_lambdas)."""
        if _is_sparse(x):
            return np.asarray(x @ self.coefs_.T) + self.intercepts_[None, :]
        x = np.ascontiguousarray(x, dtype=np.float64)
        return x @ self.coefs_.T + self.intercepts_[None, :]

    def predict(self, x) -> NDArray[np.float64]:
        """ŷ per λ: shape (n_samples, n_lambdas). Mean prediction is the
        identity here — Huber uses the identity link, the difference vs. LS
        is purely in how residuals enter the loss."""
        return self.decision_function(x)


def _check_huber_dense(x) -> None:
    if _is_sparse(x):
        raise NotImplementedError(
            "Sparse Huber path is not yet wired (PyO3 *_sparse entry pending). "
            "Convert to dense via `x.toarray()` for now."
        )


class HuberMCPRegressor(_HuberRegressorBase):
    """Huber-robust regression with MCP penalty at a single λ.

    Parameters
    ----------
    lambda_ : float, default 0.1
        Penalty strength.
    delta : float
        Huber threshold. Recommended starting point: `1.345 · σ_MAD(y)` for
        95% asymptotic efficiency at the normal (Huber, 1981).
    gamma : float, default 3.0
        MCP nonconvexity (>1). ``gamma=1e6`` is ≈ lasso.
    """

    def __init__(
        self,
        lambda_: float = 0.1,
        delta: float = 1.345,
        gamma: float = 3.0,
        *,
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.lambda_ = lambda_
        self.delta = delta
        self.gamma = gamma
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def fit(self, x, y) -> "HuberMCPRegressor":
        _check_huber_dense(x)
        common, _payload, n_features = _glm_dispatch_inputs(
            self, x, y, validate_y=_validate_y_finite, is_path=False,
        )
        common["delta"] = float(self.delta)
        common["gamma"] = self.gamma
        x_arr = common.pop("_x")
        y_arr = common.pop("_y")
        coefs, intercepts, _, info = _core.solve_huber_mcp_path(
            x_arr, y_arr, **common
        )
        self.coef_ = coefs[0]
        self.intercept_ = float(intercepts[0])
        self.info_ = info
        self.n_features_in_ = n_features
        return self


class HuberMCPPathRegressor(_HuberPathRegressorBase):
    """Huber-robust regression with MCP penalty along an entire λ-path."""

    def __init__(
        self,
        delta: float = 1.345,
        gamma: float = 3.0,
        *,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        weights: NDArray[np.float64] | None = None,
        sample_weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.delta = delta
        self.gamma = gamma
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.weights = weights
        self.sample_weights = sample_weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def fit(self, x, y) -> "HuberMCPPathRegressor":
        _check_huber_dense(x)
        common, _payload, n_features = _glm_dispatch_inputs(
            self, x, y, validate_y=_validate_y_finite, is_path=True,
        )
        common["delta"] = float(self.delta)
        common["gamma"] = self.gamma
        x_arr = common.pop("_x")
        y_arr = common.pop("_y")
        coefs, intercepts, lambdas_used, info = _core.solve_huber_mcp_path(
            x_arr, y_arr, **common
        )
        self.coefs_ = coefs
        self.intercepts_ = intercepts
        self.lambdas_ = lambdas_used
        self.info_ = info
        self.n_features_in_ = n_features
        return self


class HuberSCADRegressor(_HuberRegressorBase):
    """Huber-robust regression with SCAD penalty at a single λ."""

    def __init__(
        self,
        lambda_: float = 0.1,
        delta: float = 1.345,
        a: float = 3.7,
        *,
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.lambda_ = lambda_
        self.delta = delta
        self.a = a
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def fit(self, x, y) -> "HuberSCADRegressor":
        _check_huber_dense(x)
        common, _payload, n_features = _glm_dispatch_inputs(
            self, x, y, validate_y=_validate_y_finite, is_path=False,
        )
        common["delta"] = float(self.delta)
        common["a"] = self.a
        x_arr = common.pop("_x")
        y_arr = common.pop("_y")
        coefs, intercepts, _, info = _core.solve_huber_scad_path(
            x_arr, y_arr, **common
        )
        self.coef_ = coefs[0]
        self.intercept_ = float(intercepts[0])
        self.info_ = info
        self.n_features_in_ = n_features
        return self


class HuberSCADPathRegressor(_HuberPathRegressorBase):
    """Huber-robust regression with SCAD penalty along an entire λ-path."""

    def __init__(
        self,
        delta: float = 1.345,
        a: float = 3.7,
        *,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        weights: NDArray[np.float64] | None = None,
        sample_weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.delta = delta
        self.a = a
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.weights = weights
        self.sample_weights = sample_weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def fit(self, x, y) -> "HuberSCADPathRegressor":
        _check_huber_dense(x)
        common, _payload, n_features = _glm_dispatch_inputs(
            self, x, y, validate_y=_validate_y_finite, is_path=True,
        )
        common["delta"] = float(self.delta)
        common["a"] = self.a
        x_arr = common.pop("_x")
        y_arr = common.pop("_y")
        coefs, intercepts, lambdas_used, info = _core.solve_huber_scad_path(
            x_arr, y_arr, **common
        )
        self.coefs_ = coefs
        self.intercepts_ = intercepts
        self.lambdas_ = lambdas_used
        self.info_ = info
        self.n_features_in_ = n_features
        return self


# ============================================================
# Graphical models (M11): sparse precision matrix estimation.
# ============================================================
#
# These follow sklearn.covariance.GraphicalLasso conventions: each
# estimator's `fit(X)` accepts either raw `X (n, p)` or a precomputed
# `(p, p)` symmetric covariance, sniffed by shape + symmetry. Fitted
# attributes: `precision_`, `covariance_`, `info_`, `n_features_in_`.
# Joint variants take a list of such arrays and expose `precisions_`
# (covariances are not exposed for joint — ADMM doesn't return them).
#
# Why MCP/SCAD on edges: closes the well-known shrinkage-bias gap of
# the standard L1 glasso used in network psychometrics (Borsboom/
# Epskamp lineage), sociology, finance.


def _is_symmetric(x: NDArray[np.float64], atol: float = 1e-8) -> bool:
    return bool(np.allclose(x, x.T, atol=atol))


def _to_covariance(x: NDArray[np.float64], assume_centered: bool = False) -> NDArray[np.float64]:
    """Resolve `fit(X)`'s polymorphic input. Square + symmetric ⇒
    treat as precomputed covariance. Otherwise compute the empirical
    covariance with the MLE `1/n` denominator — the convention used
    by sklearn's GraphicalLasso, R's `glasso`, and Friedman/Hastie/
    Tibshirani 2008. `assume_centered=True` skips the mean
    subtraction."""
    if x.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {x.shape}")
    n, p = x.shape
    if n == p and _is_symmetric(x):
        return x
    if assume_centered:
        return (x.T @ x) / n
    xc = x - x.mean(axis=0, keepdims=True)
    return (xc.T @ xc) / max(n, 1)


def _validate_edge_weights(
    edge_weights: NDArray[np.float64] | None, p: int
) -> NDArray[np.float64] | None:
    if edge_weights is None:
        return None
    w = np.ascontiguousarray(edge_weights, dtype=np.float64)
    if w.shape != (p, p):
        raise ValueError(
            f"edge_weights must have shape ({p}, {p}), got {w.shape}"
        )
    if not _is_symmetric(w):
        raise ValueError("edge_weights must be symmetric")
    if np.any(w < 0):
        raise ValueError("edge_weights must be non-negative")
    return w


class _GraphicalEstimatorBase(BaseEstimator):
    precision_: NDArray[np.float64]
    covariance_: NDArray[np.float64]
    info_: dict[str, Any]
    n_features_in_: int
    # Subclasses override `_solver` with one of `_core.solve_glasso_{lasso,mcp,scad}`,
    # which have different signatures (extra `gamma` / `a` positional). Typed
    # loosely so mypy doesn't flag the override variance — the actual signature
    # is enforced by each subclass's `_solver_kwargs`.
    _solver: Callable[..., Any]


class GraphicalLasso(_GraphicalEstimatorBase):
    """L1-penalised sparse inverse covariance (graphical lasso).

    Estimates a sparse precision matrix ``Θ = Σ⁻¹`` by solving the
    L1-penalised Gaussian log-likelihood with off-diagonal-only
    weights, via the Friedman/Hastie/Tibshirani 2008 block-coordinate-descent
    algorithm. Zeros in `Θ` correspond to conditional independence
    between variables — the standard interpretation of a Gaussian
    graphical model.

    Parameters
    ----------
    alpha : float, default 0.1
        L1 regularisation strength. Larger → sparser graph.
    edge_weights : (p, p) array or None
        Per-edge weights (zero diagonal ignored). Adaptive glasso falls
        out for free: set `edge_weights[i, j] = 1 / |Θ̂_init[i, j]|`.
    assume_centered : bool, default False
        Skip mean-centering when computing the empirical covariance.
    diag_offset : float or None
        Added to the diagonal of `S` at initialisation; numerical
        safety to keep `W` positive definite. Defaults to `alpha` if
        `None`, matching sklearn's convention.
    max_iter : int, default 100
    tol : float, default 1e-4
    inner_max_iter : int, default 200
    inner_tol : float, default 1e-6

    Attributes
    ----------
    precision_ : ndarray of shape (p, p)
        Estimated precision matrix `Θ̂`.
    covariance_ : ndarray of shape (p, p)
        Working covariance estimate `Ŵ` at convergence.
    info_ : dict
    n_features_in_ : int
    """

    _solver: Callable[..., Any] = staticmethod(_core.solve_glasso_lasso)

    def __init__(
        self,
        alpha: float = 0.1,
        *,
        edge_weights: NDArray[np.float64] | None = None,
        assume_centered: bool = False,
        diag_offset: float | None = None,
        max_iter: int = 100,
        tol: float = 1e-4,
        inner_max_iter: int = 200,
        inner_tol: float = 1e-6,
    ) -> None:
        self.alpha = alpha
        self.edge_weights = edge_weights
        self.assume_centered = assume_centered
        self.diag_offset = diag_offset
        self.max_iter = max_iter
        self.tol = tol
        self.inner_max_iter = inner_max_iter
        self.inner_tol = inner_tol

    def _solver_kwargs(self) -> dict[str, Any]:
        return {"lambda_": self.alpha}

    def fit(self, X) -> "GraphicalLasso":
        x = np.ascontiguousarray(X, dtype=np.float64)
        s = _to_covariance(x, assume_centered=self.assume_centered)
        p = s.shape[0]
        ew = _validate_edge_weights(self.edge_weights, p)
        diag_offset = self.diag_offset if self.diag_offset is not None else self.alpha
        kwargs = self._solver_kwargs()
        precision, covariance, info = self._solver(
            s,
            **kwargs,
            edge_weights=ew,
            diag_offset=diag_offset,
            max_outer_iter=self.max_iter,
            outer_tol=self.tol,
            inner_max_iter=self.inner_max_iter,
            inner_tol=self.inner_tol,
        )
        self.precision_ = precision
        self.covariance_ = covariance
        self.info_ = info
        self.n_features_in_ = p
        return self


class GraphicalMCP(GraphicalLasso):
    """MCP-penalised sparse inverse covariance — reduces the shrinkage
    bias of the standard L1 glasso by transitioning to identity above
    `gamma · alpha`. Same `fit` API and fitted attributes as
    :class:`GraphicalLasso`.

    Parameters
    ----------
    gamma : float, default 3.0
        MCP nonconvexity parameter (`> 1`). `gamma → ∞` recovers
        :class:`GraphicalLasso`.
    """

    _solver = staticmethod(_core.solve_glasso_mcp)

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 3.0,
        *,
        edge_weights: NDArray[np.float64] | None = None,
        assume_centered: bool = False,
        diag_offset: float | None = None,
        max_iter: int = 100,
        tol: float = 1e-4,
        inner_max_iter: int = 200,
        inner_tol: float = 1e-6,
    ) -> None:
        super().__init__(
            alpha=alpha,
            edge_weights=edge_weights,
            assume_centered=assume_centered,
            diag_offset=diag_offset,
            max_iter=max_iter,
            tol=tol,
            inner_max_iter=inner_max_iter,
            inner_tol=inner_tol,
        )
        self.gamma = gamma

    def _solver_kwargs(self) -> dict[str, Any]:
        return {"lambda_": self.alpha, "gamma": self.gamma}


class GraphicalSCAD(GraphicalLasso):
    """SCAD-penalised sparse inverse covariance. Same `fit` API and
    fitted attributes as :class:`GraphicalLasso`.

    Parameters
    ----------
    a : float, default 3.7
        SCAD shape parameter (`> 2`). `a → ∞` recovers
        :class:`GraphicalLasso`.
    """

    _solver = staticmethod(_core.solve_glasso_scad)

    def __init__(
        self,
        alpha: float = 0.1,
        a: float = 3.7,
        *,
        edge_weights: NDArray[np.float64] | None = None,
        assume_centered: bool = False,
        diag_offset: float | None = None,
        max_iter: int = 100,
        tol: float = 1e-4,
        inner_max_iter: int = 200,
        inner_tol: float = 1e-6,
    ) -> None:
        super().__init__(
            alpha=alpha,
            edge_weights=edge_weights,
            assume_centered=assume_centered,
            diag_offset=diag_offset,
            max_iter=max_iter,
            tol=tol,
            inner_max_iter=inner_max_iter,
            inner_tol=inner_tol,
        )
        self.a = a

    def _solver_kwargs(self) -> dict[str, Any]:
        return {"lambda_": self.alpha, "a": self.a}


class _JointGraphicalBase(BaseEstimator):
    precisions_: list[NDArray[np.float64]]
    info_: dict[str, Any]
    n_features_in_: int
    n_populations_: int
    # Loosely typed for the same reason as `_GraphicalEstimatorBase._solver`:
    # subclasses point this at pyfunctions with different signatures.
    _solver: Callable[..., Any]


class JointGraphicalLasso(_JointGraphicalBase):
    """Joint graphical lasso across `K` related populations
    (Danaher–Wang–Witten 2014, group form).

    Estimates `K` precision matrices `Θ̂^(1), …, Θ̂^(K)` simultaneously
    with a group penalty coupling the *same* edge across populations.
    The coupling parameter `lambda_2` controls how much the edges
    align across populations: `lambda_2 = 0` decouples to `K`
    independent fits; large `lambda_2` collapses to identical
    estimates.

    Parameters
    ----------
    lambda_2 : float, default 0.1
        Across-population coupling strength.
    edge_weights : (p, p) array or None
        Per-edge coupling weights.
    assume_centered : bool, default False
    rho : float, default 1.0
        ADMM penalty parameter.
    diag_offset : float, default 0.0
    max_iter : int, default 200
    primal_tol : float, default 1e-5
    dual_tol : float, default 1e-5

    Attributes
    ----------
    precisions_ : list of K ndarrays of shape (p, p)
    info_ : dict
    n_features_in_ : int
    n_populations_ : int

    Notes
    -----
    First ADMM-based solver in skein. Solves the *group form* of joint
    glasso; the fused form (TV across populations) is not yet
    implemented.
    """

    _solver: Callable[..., Any] = staticmethod(_core.solve_joint_glasso_lasso)

    def __init__(
        self,
        lambda_2: float = 0.1,
        *,
        edge_weights: NDArray[np.float64] | None = None,
        assume_centered: bool = False,
        rho: float = 1.0,
        diag_offset: float = 0.0,
        max_iter: int = 200,
        primal_tol: float = 1e-5,
        dual_tol: float = 1e-5,
    ) -> None:
        self.lambda_2 = lambda_2
        self.edge_weights = edge_weights
        self.assume_centered = assume_centered
        self.rho = rho
        self.diag_offset = diag_offset
        self.max_iter = max_iter
        self.primal_tol = primal_tol
        self.dual_tol = dual_tol

    def _solver_kwargs(self) -> dict[str, Any]:
        return {"lambda_": self.lambda_2}

    def fit(self, Xs) -> "JointGraphicalLasso":
        if not isinstance(Xs, (list, tuple)) or len(Xs) == 0:
            raise ValueError(
                "Xs must be a non-empty list/tuple of arrays (one per population)"
            )
        covs: list[NDArray[np.float64]] = []
        ns: list[float] = []
        p = None
        for k, x in enumerate(Xs):
            xa = np.ascontiguousarray(x, dtype=np.float64)
            if xa.ndim != 2:
                raise ValueError(f"Xs[{k}] must be 2D, got shape {xa.shape}")
            s = _to_covariance(xa, assume_centered=self.assume_centered)
            if p is None:
                p = s.shape[0]
            elif s.shape[0] != p:
                raise ValueError(
                    f"all populations must share the same number of features; "
                    f"got {p} and {s.shape[0]}"
                )
            covs.append(s)
            # n_k: the number of samples that produced S_k. For
            # precomputed covariance, default to `s.shape[0]` so the
            # log-lik scaling is reasonable; users can override by
            # passing raw `X (n_k, p)` so we see the real n_k.
            n_k = xa.shape[0] if xa.shape != s.shape else s.shape[0]
            ns.append(float(n_k))
        assert p is not None
        ew = _validate_edge_weights(self.edge_weights, p)
        kwargs = self._solver_kwargs()
        precisions, info = self._solver(
            covs,
            ns,
            **kwargs,
            edge_weights=ew,
            rho=self.rho,
            diag_offset=self.diag_offset,
            max_iter=self.max_iter,
            primal_tol=self.primal_tol,
            dual_tol=self.dual_tol,
        )
        self.precisions_ = list(precisions)
        self.info_ = info
        self.n_features_in_ = p
        self.n_populations_ = len(precisions)
        return self


class JointGraphicalMCP(JointGraphicalLasso):
    """Joint graphical lasso with group MCP coupling. Same `fit` API
    and fitted attributes as :class:`JointGraphicalLasso`.

    Parameters
    ----------
    gamma : float, default 3.0
        Group-MCP nonconvexity parameter.
    """

    _solver = staticmethod(_core.solve_joint_glasso_mcp)

    def __init__(
        self,
        lambda_2: float = 0.1,
        gamma: float = 3.0,
        *,
        edge_weights: NDArray[np.float64] | None = None,
        assume_centered: bool = False,
        rho: float = 1.0,
        diag_offset: float = 0.0,
        max_iter: int = 200,
        primal_tol: float = 1e-5,
        dual_tol: float = 1e-5,
    ) -> None:
        super().__init__(
            lambda_2=lambda_2,
            edge_weights=edge_weights,
            assume_centered=assume_centered,
            rho=rho,
            diag_offset=diag_offset,
            max_iter=max_iter,
            primal_tol=primal_tol,
            dual_tol=dual_tol,
        )
        self.gamma = gamma

    def _solver_kwargs(self) -> dict[str, Any]:
        return {"lambda_": self.lambda_2, "gamma": self.gamma}
