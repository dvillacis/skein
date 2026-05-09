"""sklearn-compatible multi-task LS estimators (M7.1).

All take 2D `Y` of shape `(n, K)` and produce `coef_` of shape `(K, p)`,
`intercept_` of shape `(K,)`, `predict(X) → (n, K)` — matching sklearn's
`MultiTaskLasso` convention.

The Rust core returns coefficients in a row-major bvec layout
`bvec[jK+k] = B[j, k]` of shape `(n_lambdas, p*K)`. We reshape to
`(n_lambdas, p, K)` then transpose to `(n_lambdas, K, p)` here so the
sklearn-facing surface is the conventional shape.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, RegressorMixin

from skein_glm import _core


def _is_sparse(x) -> bool:
    """True if `x` is a scipy.sparse matrix-like object. scipy is imported
    lazily so dense-only users don't pay the import cost."""
    try:
        from scipy import sparse  # type: ignore[import-untyped]
    except ImportError:
        return False
    return sparse.issparse(x)


def _validate_xy_multitask(x, y):
    """Coerce + shape-check the multi-task inputs. Dispatches between
    dense ndarray and scipy.sparse CSC.

    Returns `(payload, n, p, K)` where `payload` is a dict with either
    `{"x": dense_ndarray, "y": Y_ndarray}` or `{"sparse": (data, indices,
    indptr, n_rows, n_cols), "y": Y_ndarray}`."""
    y_arr = np.ascontiguousarray(y, dtype=np.float64)
    if y_arr.ndim != 2:
        raise ValueError(
            f"Y must be 2D (shape (n, K)) for multi-task; got shape {y_arr.shape}"
        )
    k = y_arr.shape[1]
    if k < 1:
        raise ValueError("Y must have at least one task (column)")

    if _is_sparse(x):
        from scipy import sparse  # type: ignore[import-untyped]
        if not sparse.isspmatrix_csc(x):
            x = x.tocsc()
        n_rows, n_cols = x.shape
        if y_arr.shape[0] != n_rows:
            raise ValueError(
                f"Y must have {n_rows} rows (matching X); got shape {y_arr.shape}"
            )
        data = np.ascontiguousarray(x.data, dtype=np.float64)
        indices = np.ascontiguousarray(x.indices, dtype=np.int64)
        indptr = np.ascontiguousarray(x.indptr, dtype=np.int64)
        return (
            {"sparse": (data, indices, indptr, int(n_rows), int(n_cols)), "y": y_arr},
            int(n_rows),
            int(n_cols),
            k,
        )

    x_arr = np.ascontiguousarray(x, dtype=np.float64)
    if x_arr.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {x_arr.shape}")
    n, p = x_arr.shape
    if y_arr.shape[0] != n:
        raise ValueError(
            f"Y must have {n} rows (matching X); got shape {y_arr.shape}"
        )
    return ({"x": x_arr, "y": y_arr}, n, p, k)


def _multitask_dispatch(payload, kwargs, dense_fn, sparse_fn):
    """Call `dense_fn` or `sparse_fn` depending on `payload`'s shape."""
    if "sparse" in payload:
        data, indices, indptr, n_rows, n_cols = payload["sparse"]
        return sparse_fn(data, indices, indptr, n_rows, n_cols, payload["y"], **kwargs)
    return dense_fn(payload["x"], payload["y"], **kwargs)


def _bvec_to_coefs(betas: NDArray[np.float64], p: int, k: int) -> NDArray[np.float64]:
    """Reshape Rust output `(n_lambdas, p*K)` row-major bvec to sklearn-style
    `(n_lambdas, K, p)` with `coefs[lam_idx, task, feature] = B[feature, task]`."""
    n_lambdas = betas.shape[0]
    # Row-major bvec: bvec[j*K + k] = B[j, k]. Reshape to (n_lambdas, p, K),
    # then move task axis to the front of the per-lambda coef.
    return betas.reshape(n_lambdas, p, k).transpose(0, 2, 1).copy()


# =========================================================================
# Multi-task lasso (convex)
# =========================================================================


class MultiTaskLassoRegressor(BaseEstimator, RegressorMixin):
    """Multi-task lasso at a single λ. Penalty is
    `λ Σ_j w_j ‖B[j, :]‖_2` — feature `j` is selected jointly across all
    tasks. `coef_` has shape `(K, p)`, `intercept_` shape `(K,)`,
    `predict(X)` returns `(n, K)`."""

    info_: dict[str, Any]
    coef_: NDArray[np.float64]
    intercept_: NDArray[np.float64]
    n_features_in_: int
    n_tasks_: int

    def __init__(
        self,
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
        self.lambda_ = lambda_
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.screening = screening
        self.acceleration = acceleration
        self.parallel = parallel

    def fit(self, x, y) -> "MultiTaskLassoRegressor":
        payload, _n, p, k = _validate_xy_multitask(x, y)
        w = (
            np.ascontiguousarray(self.weights, dtype=np.float64)
            if self.weights is not None
            else None
        )
        kwargs = dict(
            lambdas=np.array([self.lambda_], dtype=np.float64),
            weights=w,
            max_iter=self.max_iter,
            tol=self.tol,
            screening=self.screening,
            acceleration=self.acceleration,
            parallel=self.parallel,
            fit_intercept=self.fit_intercept,
            standardize_x=self.standardize,
        )
        coefs, intercepts, _, info = _multitask_dispatch(
            payload,
            kwargs,
            _core.solve_multitask_lasso_ls_path,
            _core.solve_multitask_lasso_ls_path_sparse,
        )
        coefs_3d = _bvec_to_coefs(coefs, p, k)
        self.coef_ = coefs_3d[0]                   # (K, p)
        self.intercept_ = intercepts[0].copy()     # (K,)
        self.info_ = info
        self.n_features_in_ = p
        self.n_tasks_ = k
        return self

    def predict(self, x) -> NDArray[np.float64]:
        if _is_sparse(x):
            if x.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"X must be 2D with {self.n_features_in_} features; "
                    f"got shape {x.shape}"
                )
            pred = x @ self.coef_.T
            if hasattr(pred, "toarray"):
                pred = pred.toarray()
            return np.asarray(pred) + self.intercept_
        x_arr = np.ascontiguousarray(x, dtype=np.float64)
        if x_arr.ndim != 2 or x_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X must be 2D with {self.n_features_in_} features; "
                f"got shape {x_arr.shape}"
            )
        return x_arr @ self.coef_.T + self.intercept_


class MultiTaskLassoPathRegressor(BaseEstimator):
    """Multi-task lasso along a λ-path with warm starts. After `fit`,
    `coefs_` has shape `(n_lambdas, K, p)`, `intercepts_` shape
    `(n_lambdas, K)`, `lambdas_` shape `(n_lambdas,)`."""

    info_: dict[str, Any]
    coefs_: NDArray[np.float64]
    intercepts_: NDArray[np.float64]
    lambdas_: NDArray[np.float64]
    n_features_in_: int
    n_tasks_: int

    def __init__(
        self,
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

    def fit(self, x, y) -> "MultiTaskLassoPathRegressor":
        payload, _n, p, k = _validate_xy_multitask(x, y)
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
        kwargs = dict(
            lambdas=lams,
            n_lambdas=self.n_lambdas,
            lambda_min_ratio=self.lambda_min_ratio,
            weights=w,
            max_iter=self.max_iter,
            tol=self.tol,
            screening=self.screening,
            acceleration=self.acceleration,
            parallel=self.parallel,
            fit_intercept=self.fit_intercept,
            standardize_x=self.standardize,
        )
        coefs, intercepts, lambdas_used, info = _multitask_dispatch(
            payload,
            kwargs,
            _core.solve_multitask_lasso_ls_path,
            _core.solve_multitask_lasso_ls_path_sparse,
        )
        self.coefs_ = _bvec_to_coefs(coefs, p, k)
        self.intercepts_ = intercepts.copy()
        self.lambdas_ = lambdas_used
        self.info_ = info
        self.n_features_in_ = p
        self.n_tasks_ = k
        return self


# =========================================================================
# Multi-task MCP (LLA-wrapped, non-convex)
# =========================================================================


class MultiTaskMCPRegressor(BaseEstimator, RegressorMixin):
    """Multi-task MCP at a single λ via LLA outer loop. The non-convex
    row-group MCP penalty is linearized to a weighted multi-task lasso at
    each LLA outer iteration."""

    info_: dict[str, Any]
    coef_: NDArray[np.float64]
    intercept_: NDArray[np.float64]
    n_features_in_: int
    n_tasks_: int

    def __init__(
        self,
        lambda_: float = 0.1,
        gamma: float = 3.0,
        *,
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        screening: str = "strong",
        acceleration: int | None = 5,
        parallel: bool = False,
    ) -> None:
        self.lambda_ = lambda_
        self.gamma = gamma
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.max_outer = max_outer
        self.outer_tol = outer_tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.screening = screening
        self.acceleration = acceleration
        self.parallel = parallel

    def fit(self, x, y) -> "MultiTaskMCPRegressor":
        payload, _n, p, k = _validate_xy_multitask(x, y)
        w = (
            np.ascontiguousarray(self.weights, dtype=np.float64)
            if self.weights is not None
            else None
        )
        kwargs = dict(
            gamma=self.gamma,
            lambdas=np.array([self.lambda_], dtype=np.float64),
            weights=w,
            max_iter=self.max_iter,
            tol=self.tol,
            screening=self.screening,
            acceleration=self.acceleration,
            parallel=self.parallel,
            fit_intercept=self.fit_intercept,
            max_outer=self.max_outer,
            outer_tol=self.outer_tol,
        )
        coefs, intercepts, _, info = _multitask_dispatch(
            payload,
            kwargs,
            _core.solve_multitask_mcp_ls_path,
            _core.solve_multitask_mcp_ls_path_sparse,
        )
        coefs_3d = _bvec_to_coefs(coefs, p, k)
        self.coef_ = coefs_3d[0]
        self.intercept_ = intercepts[0].copy()
        self.info_ = info
        self.n_features_in_ = p
        self.n_tasks_ = k
        return self

    def predict(self, x) -> NDArray[np.float64]:
        if _is_sparse(x):
            if x.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"X must be 2D with {self.n_features_in_} features; "
                    f"got shape {x.shape}"
                )
            pred = x @ self.coef_.T
            if hasattr(pred, "toarray"):
                pred = pred.toarray()
            return np.asarray(pred) + self.intercept_
        x_arr = np.ascontiguousarray(x, dtype=np.float64)
        if x_arr.ndim != 2 or x_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X must be 2D with {self.n_features_in_} features; "
                f"got shape {x_arr.shape}"
            )
        return x_arr @ self.coef_.T + self.intercept_


class MultiTaskMCPPathRegressor(BaseEstimator):
    """Multi-task MCP along a λ-path via LLA at every λ."""

    info_: dict[str, Any]
    coefs_: NDArray[np.float64]
    intercepts_: NDArray[np.float64]
    lambdas_: NDArray[np.float64]
    n_features_in_: int
    n_tasks_: int

    def __init__(
        self,
        gamma: float = 3.0,
        *,
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
        screening: str = "strong",
        acceleration: int | None = 5,
        parallel: bool = False,
    ) -> None:
        self.gamma = gamma
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
        self.screening = screening
        self.acceleration = acceleration
        self.parallel = parallel

    def fit(self, x, y) -> "MultiTaskMCPPathRegressor":
        payload, _n, p, k = _validate_xy_multitask(x, y)
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
        kwargs = dict(
            gamma=self.gamma,
            lambdas=lams,
            n_lambdas=self.n_lambdas,
            lambda_min_ratio=self.lambda_min_ratio,
            weights=w,
            max_iter=self.max_iter,
            tol=self.tol,
            screening=self.screening,
            acceleration=self.acceleration,
            parallel=self.parallel,
            fit_intercept=self.fit_intercept,
            max_outer=self.max_outer,
            outer_tol=self.outer_tol,
        )
        coefs, intercepts, lambdas_used, info = _multitask_dispatch(
            payload,
            kwargs,
            _core.solve_multitask_mcp_ls_path,
            _core.solve_multitask_mcp_ls_path_sparse,
        )
        self.coefs_ = _bvec_to_coefs(coefs, p, k)
        self.intercepts_ = intercepts.copy()
        self.lambdas_ = lambdas_used
        self.info_ = info
        self.n_features_in_ = p
        self.n_tasks_ = k
        return self


# =========================================================================
# Multi-task SCAD (LLA-wrapped, non-convex)
# =========================================================================


class MultiTaskSCADRegressor(BaseEstimator, RegressorMixin):
    """Multi-task SCAD at a single λ via LLA outer loop. SCAD shape `a > 2`
    (default 3.7 — Fan & Li's recommendation)."""

    info_: dict[str, Any]
    coef_: NDArray[np.float64]
    intercept_: NDArray[np.float64]
    n_features_in_: int
    n_tasks_: int

    def __init__(
        self,
        lambda_: float = 0.1,
        a: float = 3.7,
        *,
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        screening: str = "strong",
        acceleration: int | None = 5,
        parallel: bool = False,
    ) -> None:
        self.lambda_ = lambda_
        self.a = a
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.max_outer = max_outer
        self.outer_tol = outer_tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.screening = screening
        self.acceleration = acceleration
        self.parallel = parallel

    def fit(self, x, y) -> "MultiTaskSCADRegressor":
        payload, _n, p, k = _validate_xy_multitask(x, y)
        w = (
            np.ascontiguousarray(self.weights, dtype=np.float64)
            if self.weights is not None
            else None
        )
        kwargs = dict(
            a=self.a,
            lambdas=np.array([self.lambda_], dtype=np.float64),
            weights=w,
            max_iter=self.max_iter,
            tol=self.tol,
            screening=self.screening,
            acceleration=self.acceleration,
            parallel=self.parallel,
            fit_intercept=self.fit_intercept,
            max_outer=self.max_outer,
            outer_tol=self.outer_tol,
        )
        coefs, intercepts, _, info = _multitask_dispatch(
            payload,
            kwargs,
            _core.solve_multitask_scad_ls_path,
            _core.solve_multitask_scad_ls_path_sparse,
        )
        coefs_3d = _bvec_to_coefs(coefs, p, k)
        self.coef_ = coefs_3d[0]
        self.intercept_ = intercepts[0].copy()
        self.info_ = info
        self.n_features_in_ = p
        self.n_tasks_ = k
        return self

    def predict(self, x) -> NDArray[np.float64]:
        if _is_sparse(x):
            if x.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"X must be 2D with {self.n_features_in_} features; "
                    f"got shape {x.shape}"
                )
            pred = x @ self.coef_.T
            if hasattr(pred, "toarray"):
                pred = pred.toarray()
            return np.asarray(pred) + self.intercept_
        x_arr = np.ascontiguousarray(x, dtype=np.float64)
        if x_arr.ndim != 2 or x_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X must be 2D with {self.n_features_in_} features; "
                f"got shape {x_arr.shape}"
            )
        return x_arr @ self.coef_.T + self.intercept_


class MultiTaskSCADPathRegressor(BaseEstimator):
    """Multi-task SCAD along a λ-path via LLA at every λ."""

    info_: dict[str, Any]
    coefs_: NDArray[np.float64]
    intercepts_: NDArray[np.float64]
    lambdas_: NDArray[np.float64]
    n_features_in_: int
    n_tasks_: int

    def __init__(
        self,
        a: float = 3.7,
        *,
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
        screening: str = "strong",
        acceleration: int | None = 5,
        parallel: bool = False,
    ) -> None:
        self.a = a
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
        self.screening = screening
        self.acceleration = acceleration
        self.parallel = parallel

    def fit(self, x, y) -> "MultiTaskSCADPathRegressor":
        payload, _n, p, k = _validate_xy_multitask(x, y)
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
        kwargs = dict(
            a=self.a,
            lambdas=lams,
            n_lambdas=self.n_lambdas,
            lambda_min_ratio=self.lambda_min_ratio,
            weights=w,
            max_iter=self.max_iter,
            tol=self.tol,
            screening=self.screening,
            acceleration=self.acceleration,
            parallel=self.parallel,
            fit_intercept=self.fit_intercept,
            max_outer=self.max_outer,
            outer_tol=self.outer_tol,
        )
        coefs, intercepts, lambdas_used, info = _multitask_dispatch(
            payload,
            kwargs,
            _core.solve_multitask_scad_ls_path,
            _core.solve_multitask_scad_ls_path_sparse,
        )
        self.coefs_ = _bvec_to_coefs(coefs, p, k)
        self.intercepts_ = intercepts.copy()
        self.lambdas_ = lambdas_used
        self.info_ = info
        self.n_features_in_ = p
        self.n_tasks_ = k
        return self


# =========================================================================
# Multi-task elastic net (convex)
# =========================================================================


class MultiTaskElasticNetRegressor(BaseEstimator, RegressorMixin):
    """Multi-task elastic net at a single λ. Convex per-row penalty
    `α λ w_j ‖B[j,:]‖₂ + (1-α) λ w_j ‖B[j,:]‖₂² / 2`. `α=1` reduces to
    multi-task lasso; `α=0` is per-row ridge."""

    info_: dict[str, Any]
    coef_: NDArray[np.float64]
    intercept_: NDArray[np.float64]
    n_features_in_: int
    n_tasks_: int

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
        parallel: bool = False,
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
        self.parallel = parallel

    def fit(self, x, y) -> "MultiTaskElasticNetRegressor":
        payload, _n, p, k = _validate_xy_multitask(x, y)
        w = (
            np.ascontiguousarray(self.weights, dtype=np.float64)
            if self.weights is not None
            else None
        )
        kwargs = dict(
            alpha=self.alpha,
            lambdas=np.array([self.lambda_], dtype=np.float64),
            weights=w,
            max_iter=self.max_iter,
            tol=self.tol,
            screening=self.screening,
            acceleration=self.acceleration,
            parallel=self.parallel,
            fit_intercept=self.fit_intercept,
            standardize_x=self.standardize,
        )
        coefs, intercepts, _, info = _multitask_dispatch(
            payload,
            kwargs,
            _core.solve_multitask_elastic_net_ls_path,
            _core.solve_multitask_elastic_net_ls_path_sparse,
        )
        coefs_3d = _bvec_to_coefs(coefs, p, k)
        self.coef_ = coefs_3d[0]
        self.intercept_ = intercepts[0].copy()
        self.info_ = info
        self.n_features_in_ = p
        self.n_tasks_ = k
        return self

    def predict(self, x) -> NDArray[np.float64]:
        if _is_sparse(x):
            if x.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"X must be 2D with {self.n_features_in_} features; "
                    f"got shape {x.shape}"
                )
            pred = x @ self.coef_.T
            if hasattr(pred, "toarray"):
                pred = pred.toarray()
            return np.asarray(pred) + self.intercept_
        x_arr = np.ascontiguousarray(x, dtype=np.float64)
        if x_arr.ndim != 2 or x_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X must be 2D with {self.n_features_in_} features; "
                f"got shape {x_arr.shape}"
            )
        return x_arr @ self.coef_.T + self.intercept_


class MultiTaskElasticNetPathRegressor(BaseEstimator):
    """Multi-task elastic net along a λ-path with warm starts."""

    info_: dict[str, Any]
    coefs_: NDArray[np.float64]
    intercepts_: NDArray[np.float64]
    lambdas_: NDArray[np.float64]
    n_features_in_: int
    n_tasks_: int

    def __init__(
        self,
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

    def fit(self, x, y) -> "MultiTaskElasticNetPathRegressor":
        payload, _n, p, k = _validate_xy_multitask(x, y)
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
        kwargs = dict(
            alpha=self.alpha,
            lambdas=lams,
            n_lambdas=self.n_lambdas,
            lambda_min_ratio=self.lambda_min_ratio,
            weights=w,
            max_iter=self.max_iter,
            tol=self.tol,
            screening=self.screening,
            acceleration=self.acceleration,
            parallel=self.parallel,
            fit_intercept=self.fit_intercept,
            standardize_x=self.standardize,
        )
        coefs, intercepts, lambdas_used, info = _multitask_dispatch(
            payload,
            kwargs,
            _core.solve_multitask_elastic_net_ls_path,
            _core.solve_multitask_elastic_net_ls_path_sparse,
        )
        self.coefs_ = _bvec_to_coefs(coefs, p, k)
        self.intercepts_ = intercepts.copy()
        self.lambdas_ = lambdas_used
        self.info_ = info
        self.n_features_in_ = p
        self.n_tasks_ = k
        return self


# =========================================================================
# Multi-task path CV
# =========================================================================


class _MultiTaskPathCVBase(BaseEstimator, RegressorMixin):
    """Shared K-fold CV machinery for multi-task path estimators.

    Subclasses implement `_make_base_path(**overrides)` to return the
    underlying path estimator. We score by mean MSE averaged across tasks
    (lower-is-better)."""

    info_: dict[str, Any]
    cv_scores_: NDArray[np.float64]
    cv_mean_scores_: NDArray[np.float64]
    cv_std_scores_: NDArray[np.float64]
    lambdas_: NDArray[np.float64]
    lambda_best_: float
    coef_: NDArray[np.float64]
    intercept_: NDArray[np.float64]
    n_features_in_: int
    n_tasks_: int

    def _make_base_path(self, **overrides):  # pragma: no cover - abstract
        raise NotImplementedError

    def fit(self, x, y) -> "_MultiTaskPathCVBase":
        from sklearn.model_selection import KFold

        # Validate up-front (also normalizes y to 2D float64) and capture
        # shape; we still pass `x` unchanged so the underlying estimator
        # can dispatch dense vs sparse on its own.
        _payload, n, p, k = _validate_xy_multitask(x, y)
        y_arr = np.ascontiguousarray(y, dtype=np.float64)

        # CSC is bad at row indexing; convert once to CSR for the fold loop.
        if _is_sparse(x):
            from scipy import sparse  # type: ignore[import-untyped]
            x_for_indexing = x.tocsr() if not sparse.isspmatrix_csr(x) else x
        else:
            x_for_indexing = np.ascontiguousarray(x, dtype=np.float64)

        # Full-data fit: provides the auto λ-grid and the final refit.
        full = self._make_base_path().fit(x, y_arr)
        lambdas = full.lambdas_

        cv = self.cv  # type: ignore[attr-defined]
        random_state = self.random_state  # type: ignore[attr-defined]
        if isinstance(cv, int):
            splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
        else:
            splitter = cv

        n_lambdas = lambdas.shape[0]
        # `splitter.split` doesn't really care about feature width — pass
        # an ndarray of the right row count for both dense and sparse.
        split_input = np.zeros((n, 1)) if _is_sparse(x) else x_for_indexing
        scores = np.full((splitter.get_n_splits(split_input), n_lambdas), np.nan)

        for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(split_input)):
            x_tr = x_for_indexing[train_idx]
            x_te = x_for_indexing[test_idx]
            y_tr = y_arr[train_idx]
            y_te = y_arr[test_idx]
            fold = self._make_base_path(lambdas=lambdas).fit(x_tr, y_tr)
            for lam_idx in range(n_lambdas):
                # coefs_[lam_idx] has shape (K, p); predict per-task.
                # For sparse x_te (csr), `x_te @ dense` returns ndarray.
                pred = x_te @ fold.coefs_[lam_idx].T + fold.intercepts_[lam_idx]
                if hasattr(pred, "toarray"):
                    pred = pred.toarray()
                diff = y_te - np.asarray(pred)
                scores[fold_idx, lam_idx] = float(np.mean(diff * diff))

        self.cv_scores_ = scores
        self.cv_mean_scores_ = np.nanmean(scores, axis=0)
        self.cv_std_scores_ = np.nanstd(scores, axis=0)
        best = int(np.argmin(self.cv_mean_scores_))
        self.lambdas_ = lambdas
        self.lambda_best_ = float(lambdas[best])
        self.coef_ = full.coefs_[best]
        self.intercept_ = full.intercepts_[best].copy()
        self.info_ = full.info_
        self.n_features_in_ = p
        self.n_tasks_ = k
        return self

    def predict(self, x) -> NDArray[np.float64]:
        if _is_sparse(x):
            if x.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"X must be 2D with {self.n_features_in_} features; "
                    f"got shape {x.shape}"
                )
            pred = x @ self.coef_.T
            if hasattr(pred, "toarray"):
                pred = pred.toarray()
            return np.asarray(pred) + self.intercept_
        x_arr = np.ascontiguousarray(x, dtype=np.float64)
        if x_arr.ndim != 2 or x_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X must be 2D with {self.n_features_in_} features; "
                f"got shape {x_arr.shape}"
            )
        return x_arr @ self.coef_.T + self.intercept_


class MultiTaskLassoPathCV(_MultiTaskPathCVBase):
    """K-fold CV over a multi-task lasso λ-path. Picks the λ minimizing
    mean test MSE (averaged across tasks)."""

    def __init__(
        self,
        *,
        cv: Any = 5,
        random_state: int | None = None,
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
        self.cv = cv
        self.random_state = random_state
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

    def _make_base_path(self, **overrides) -> MultiTaskLassoPathRegressor:
        kw: dict[str, Any] = dict(
            lambdas=self.lambdas,
            n_lambdas=self.n_lambdas,
            lambda_min_ratio=self.lambda_min_ratio,
            weights=self.weights,
            max_iter=self.max_iter,
            tol=self.tol,
            fit_intercept=self.fit_intercept,
            standardize=self.standardize,
            screening=self.screening,
            acceleration=self.acceleration,
            parallel=self.parallel,
        )
        kw.update(overrides)
        return MultiTaskLassoPathRegressor(**kw)


class MultiTaskMCPPathCV(_MultiTaskPathCVBase):
    """K-fold CV over a multi-task MCP λ-path (LLA outer loop)."""

    def __init__(
        self,
        gamma: float = 3.0,
        *,
        cv: Any = 5,
        random_state: int | None = None,
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
        screening: str = "strong",
        acceleration: int | None = 5,
        parallel: bool = False,
    ) -> None:
        self.gamma = gamma
        self.cv = cv
        self.random_state = random_state
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
        self.screening = screening
        self.acceleration = acceleration
        self.parallel = parallel

    def _make_base_path(self, **overrides) -> MultiTaskMCPPathRegressor:
        kw: dict[str, Any] = dict(
            gamma=self.gamma,
            lambdas=self.lambdas,
            n_lambdas=self.n_lambdas,
            lambda_min_ratio=self.lambda_min_ratio,
            weights=self.weights,
            max_iter=self.max_iter,
            tol=self.tol,
            max_outer=self.max_outer,
            outer_tol=self.outer_tol,
            fit_intercept=self.fit_intercept,
            standardize=self.standardize,
            screening=self.screening,
            acceleration=self.acceleration,
            parallel=self.parallel,
        )
        kw.update(overrides)
        return MultiTaskMCPPathRegressor(**kw)


class MultiTaskSCADPathCV(_MultiTaskPathCVBase):
    """K-fold CV over a multi-task SCAD λ-path (LLA outer loop)."""

    def __init__(
        self,
        a: float = 3.7,
        *,
        cv: Any = 5,
        random_state: int | None = None,
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
        screening: str = "strong",
        acceleration: int | None = 5,
        parallel: bool = False,
    ) -> None:
        self.a = a
        self.cv = cv
        self.random_state = random_state
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
        self.screening = screening
        self.acceleration = acceleration
        self.parallel = parallel

    def _make_base_path(self, **overrides) -> MultiTaskSCADPathRegressor:
        kw: dict[str, Any] = dict(
            a=self.a,
            lambdas=self.lambdas,
            n_lambdas=self.n_lambdas,
            lambda_min_ratio=self.lambda_min_ratio,
            weights=self.weights,
            max_iter=self.max_iter,
            tol=self.tol,
            max_outer=self.max_outer,
            outer_tol=self.outer_tol,
            fit_intercept=self.fit_intercept,
            standardize=self.standardize,
            screening=self.screening,
            acceleration=self.acceleration,
            parallel=self.parallel,
        )
        kw.update(overrides)
        return MultiTaskSCADPathRegressor(**kw)


class MultiTaskElasticNetPathCV(_MultiTaskPathCVBase):
    """K-fold CV over a multi-task elastic-net λ-path."""

    def __init__(
        self,
        alpha: float = 0.5,
        *,
        cv: Any = 5,
        random_state: int | None = None,
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
        self.alpha = alpha
        self.cv = cv
        self.random_state = random_state
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

    def _make_base_path(self, **overrides) -> MultiTaskElasticNetPathRegressor:
        kw: dict[str, Any] = dict(
            alpha=self.alpha,
            lambdas=self.lambdas,
            n_lambdas=self.n_lambdas,
            lambda_min_ratio=self.lambda_min_ratio,
            weights=self.weights,
            max_iter=self.max_iter,
            tol=self.tol,
            fit_intercept=self.fit_intercept,
            standardize=self.standardize,
            screening=self.screening,
            acceleration=self.acceleration,
            parallel=self.parallel,
        )
        kw.update(overrides)
        return MultiTaskElasticNetPathRegressor(**kw)
