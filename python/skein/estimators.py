"""sklearn-compatible estimators built on the Rust core.

`MCPRegressor` / `SCADRegressor` are single-λ wrappers; `MCPPathRegressor` /
`SCADPathRegressor` return the entire λ-path. All four standardize and fit
an intercept by default (matching `glmnet` / `ncvreg` conventions) and
expose `coef_`, `intercept_`, and `info_` after `fit`.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, RegressorMixin

from skein import _core
from skein.mmap import MmapDesignF32, MmapDesignF64


def _is_mmap(x) -> bool:
    return isinstance(x, (MmapDesignF64, MmapDesignF32))


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
    """Least-squares regression with MCP penalty at a single λ."""

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
    """MCP regression along an entire λ-path with warm starts."""

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
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.screening = screening
        self.acceleration = acceleration

    def fit(self, x, y: NDArray[np.float64]) -> "MCPPathRegressor":
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
        if _is_mmap(x):
            if y.ndim != 1 or y.shape[0] != x.n_rows:
                raise ValueError(
                    f"y must be 1D with length {x.n_rows}, got shape {y.shape}"
                )
            entry = (
                _core.solve_mcp_ls_path_mmap_f32
                if x.dtype == "f32"
                else _core.solve_mcp_ls_path_mmap
            )
            coefs, intercepts, lambdas_used, info = entry(
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
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.screening = screening
        self.acceleration = acceleration

    def fit(self, x, y: NDArray[np.float64]) -> "SCADPathRegressor":
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
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def fit(self, x, y) -> "LogisticMCPPathRegressor":
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
            entry = (
                _core.solve_logistic_mcp_path_mmap_f32
                if x.dtype == "f32"
                else _core.solve_logistic_mcp_path_mmap
            )
            coefs, intercepts, lambdas_used, info = entry(
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
        weights: NDArray[np.float64] | None = None,
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
        weights: NDArray[np.float64] | None = None,
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
