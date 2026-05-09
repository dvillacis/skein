"""sklearn-compatible multinomial (softmax) logistic classifiers (M3.6).

K-class softmax with coefficient matrix `B ∈ ℝ^{p×K}`. The Rust core handles
the full prox-Newton outer loop wrapped around an LLA inner loop for non-
convex penalties; the surrogate at each outer iter is a multi-task LS
problem on the same `MultiTaskDesign<X>` virtual design used by `multitask`.

All estimators expose:
- `coef_` shape `(K, p)`   — `coef_[k, j] = B[j, k]` (matches sklearn).
- `intercept_` shape `(K,)`.
- `decision_function(X) → (n, K)` — η = X·coef_.T + intercept_.
- `predict_proba(X) → (n, K)` — softmax(η), rows sum to 1.
- `predict(X) → (n,)` — argmax class labels (cast back to original dtype).

The Rust output is a row-major bvec `(n_lambdas, p*K)` with
`bvec[lam, j*K + k] = B[j, k]`. We reshape to `(n_lambdas, p, K)` and
transpose to `(n_lambdas, K, p)` so the sklearn-facing surface matches
`LogisticRegression(multi_class='multinomial').coef_`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin

from skein_glm import _core


def _is_sparse(x) -> bool:
    try:
        from scipy import sparse  # type: ignore[import-untyped]
    except ImportError:
        return False
    return sparse.issparse(x)


def _encode_labels(y) -> tuple[NDArray[np.float64], NDArray]:
    """Convert arbitrary labels to integer codes in `[0, K)`. Returns
    `(codes_float64, classes_)` where `classes_` is the sorted unique array
    that lets us decode predictions back to the original dtype."""
    y_arr = np.asarray(y)
    classes = np.unique(y_arr)
    if classes.shape[0] < 2:
        raise ValueError(
            f"multinomial classification requires ≥ 2 distinct classes; got {classes.shape[0]}"
        )
    codes = np.searchsorted(classes, y_arr).astype(np.float64)
    return codes, classes


def _validate_xy_multinomial(x, y):
    """Coerce and shape-check inputs. Returns `(payload, n, p, codes,
    classes)` where `payload` carries either a dense ndarray or the CSC
    triple, plus the float64 label codes for the Rust core."""
    codes, classes = _encode_labels(y)
    n_y = codes.shape[0]
    if _is_sparse(x):
        from scipy import sparse  # type: ignore[import-untyped]

        if not sparse.isspmatrix_csc(x):
            x = x.tocsc()
        n_rows, n_cols = x.shape
        if n_y != n_rows:
            raise ValueError(
                f"y length {n_y} does not match X.shape[0] = {n_rows}"
            )
        data = np.ascontiguousarray(x.data, dtype=np.float64)
        indices = np.ascontiguousarray(x.indices, dtype=np.int64)
        indptr = np.ascontiguousarray(x.indptr, dtype=np.int64)
        return (
            {"sparse": (data, indices, indptr, int(n_rows), int(n_cols))},
            int(n_rows),
            int(n_cols),
            codes,
            classes,
        )
    x_arr = np.ascontiguousarray(x, dtype=np.float64)
    if x_arr.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {x_arr.shape}")
    n, p = x_arr.shape
    if n_y != n:
        raise ValueError(f"y length {n_y} does not match X.shape[0] = {n}")
    return ({"x": x_arr}, n, p, codes, classes)


def _multinomial_dispatch(payload, codes, n_classes, kwargs, dense_fn, sparse_fn):
    if "sparse" in payload:
        data, indices, indptr, n_rows, n_cols = payload["sparse"]
        return sparse_fn(
            n_rows, n_cols, data, indices, indptr, codes, n_classes, **kwargs
        )
    return dense_fn(payload["x"], codes, n_classes, **kwargs)


def _bvec_to_coefs(betas: NDArray[np.float64], p: int, k: int) -> NDArray[np.float64]:
    """Reshape Rust bvec `(n_lambdas, p*K)` (row-major `bvec[j*K+k]=B[j,k]`)
    to sklearn-style `(n_lambdas, K, p)` with `coefs[lam, k, j] = B[j, k]`."""
    n_lambdas = betas.shape[0]
    return betas.reshape(n_lambdas, p, k).transpose(0, 2, 1).copy()


def _softmax_2d(eta: NDArray[np.float64]) -> NDArray[np.float64]:
    """Stable row-wise softmax for an `(n, K)` η matrix."""
    m = eta.max(axis=1, keepdims=True)
    e = np.exp(eta - m)
    return e / e.sum(axis=1, keepdims=True)


# =========================================================================
# Mixins
# =========================================================================


class _MultinomialPredictMixin:
    """Shared `decision_function` / `predict_proba` / `predict` for fitted
    multinomial estimators that store `coef_ (K, p)`, `intercept_ (K,)`,
    `classes_ (K,)`, and `n_features_in_`."""

    coef_: NDArray[np.float64]
    intercept_: NDArray[np.float64]
    classes_: NDArray
    n_features_in_: int

    def decision_function(self, x) -> NDArray[np.float64]:
        if _is_sparse(x):
            if x.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"X must be 2D with {self.n_features_in_} features; "
                    f"got shape {x.shape}"
                )
            eta = x @ self.coef_.T
            if hasattr(eta, "toarray"):
                eta = eta.toarray()
            return np.asarray(eta) + self.intercept_
        x_arr = np.ascontiguousarray(x, dtype=np.float64)
        if x_arr.ndim != 2 or x_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X must be 2D with {self.n_features_in_} features; "
                f"got shape {x_arr.shape}"
            )
        return x_arr @ self.coef_.T + self.intercept_

    def predict_proba(self, x) -> NDArray[np.float64]:
        return _softmax_2d(self.decision_function(x))

    def predict(self, x) -> NDArray:
        eta = self.decision_function(x)
        idx = np.argmax(eta, axis=1)
        return self.classes_[idx]


# =========================================================================
# Single-λ classifiers
# =========================================================================


def _take_first(coefs_3d: NDArray[np.float64], intercepts: NDArray[np.float64]):
    return coefs_3d[0], intercepts[0].copy()


class MultinomialLassoClassifier(BaseEstimator, ClassifierMixin, _MultinomialPredictMixin):
    """Multinomial logistic regression with row-grouped lasso penalty
    `λ Σ_j w_j ‖B[j, :]‖_2` (joint feature selection across all classes).
    Convex; Böhning-bound Newton inner solve."""

    info_: dict[str, Any]
    classes_: NDArray
    coef_: NDArray[np.float64]
    intercept_: NDArray[np.float64]
    n_features_in_: int
    n_classes_: int

    def __init__(
        self,
        lambda_: float = 0.1,
        *,
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        max_outer: int = 25,
        outer_tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
    ) -> None:
        self.lambda_ = lambda_
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.max_outer = max_outer
        self.outer_tol = outer_tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.acceleration = acceleration

    def fit(self, x, y) -> "MultinomialLassoClassifier":
        payload, _n, p, codes, classes = _validate_xy_multinomial(x, y)
        k = int(classes.shape[0])
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
            acceleration=self.acceleration,
            fit_intercept=self.fit_intercept,
            standardize_x=self.standardize,
            max_outer=self.max_outer,
            outer_tol=self.outer_tol,
        )
        coefs, intercepts, _, info = _multinomial_dispatch(
            payload,
            codes,
            k,
            kwargs,
            _core.solve_multinomial_lasso_path,
            _core.solve_multinomial_lasso_path_sparse,
        )
        coefs_3d = _bvec_to_coefs(coefs, p, k)
        self.coef_, self.intercept_ = _take_first(coefs_3d, intercepts)
        self.classes_ = classes
        self.info_ = info
        self.n_features_in_ = p
        self.n_classes_ = k
        return self


class MultinomialMCPClassifier(BaseEstimator, ClassifierMixin, _MultinomialPredictMixin):
    """Multinomial logistic with row-grouped MCP penalty (non-convex via
    LLA outer loop)."""

    info_: dict[str, Any]
    classes_: NDArray
    coef_: NDArray[np.float64]
    intercept_: NDArray[np.float64]
    n_features_in_: int
    n_classes_: int

    def __init__(
        self,
        lambda_: float = 0.1,
        gamma: float = 3.0,
        *,
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        max_outer: int = 25,
        outer_tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
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
        self.acceleration = acceleration

    def fit(self, x, y) -> "MultinomialMCPClassifier":
        payload, _n, p, codes, classes = _validate_xy_multinomial(x, y)
        k = int(classes.shape[0])
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
            acceleration=self.acceleration,
            fit_intercept=self.fit_intercept,
            standardize_x=self.standardize,
            max_outer=self.max_outer,
            outer_tol=self.outer_tol,
        )
        coefs, intercepts, _, info = _multinomial_dispatch(
            payload,
            codes,
            k,
            kwargs,
            _core.solve_multinomial_mcp_path,
            _core.solve_multinomial_mcp_path_sparse,
        )
        coefs_3d = _bvec_to_coefs(coefs, p, k)
        self.coef_, self.intercept_ = _take_first(coefs_3d, intercepts)
        self.classes_ = classes
        self.info_ = info
        self.n_features_in_ = p
        self.n_classes_ = k
        return self


class MultinomialSCADClassifier(BaseEstimator, ClassifierMixin, _MultinomialPredictMixin):
    """Multinomial logistic with row-grouped SCAD penalty (non-convex via
    LLA outer loop). Default `a = 3.7` (Fan & Li recommendation)."""

    info_: dict[str, Any]
    classes_: NDArray
    coef_: NDArray[np.float64]
    intercept_: NDArray[np.float64]
    n_features_in_: int
    n_classes_: int

    def __init__(
        self,
        lambda_: float = 0.1,
        a: float = 3.7,
        *,
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        max_outer: int = 25,
        outer_tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
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
        self.acceleration = acceleration

    def fit(self, x, y) -> "MultinomialSCADClassifier":
        payload, _n, p, codes, classes = _validate_xy_multinomial(x, y)
        k = int(classes.shape[0])
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
            acceleration=self.acceleration,
            fit_intercept=self.fit_intercept,
            standardize_x=self.standardize,
            max_outer=self.max_outer,
            outer_tol=self.outer_tol,
        )
        coefs, intercepts, _, info = _multinomial_dispatch(
            payload,
            codes,
            k,
            kwargs,
            _core.solve_multinomial_scad_path,
            _core.solve_multinomial_scad_path_sparse,
        )
        coefs_3d = _bvec_to_coefs(coefs, p, k)
        self.coef_, self.intercept_ = _take_first(coefs_3d, intercepts)
        self.classes_ = classes
        self.info_ = info
        self.n_features_in_ = p
        self.n_classes_ = k
        return self


class MultinomialElasticNetClassifier(BaseEstimator, ClassifierMixin, _MultinomialPredictMixin):
    """Multinomial logistic with row-grouped elastic-net penalty
    `α λ w_j ‖B[j, :]‖₂ + (1-α) λ w_j ‖B[j, :]‖₂² / 2`. Convex; `α=1`
    reduces to row-grouped lasso, `α=0` to row-grouped ridge."""

    info_: dict[str, Any]
    classes_: NDArray
    coef_: NDArray[np.float64]
    intercept_: NDArray[np.float64]
    n_features_in_: int
    n_classes_: int

    def __init__(
        self,
        lambda_: float = 0.1,
        alpha: float = 0.5,
        *,
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        max_outer: int = 25,
        outer_tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
    ) -> None:
        self.lambda_ = lambda_
        self.alpha = alpha
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.max_outer = max_outer
        self.outer_tol = outer_tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.acceleration = acceleration

    def fit(self, x, y) -> "MultinomialElasticNetClassifier":
        payload, _n, p, codes, classes = _validate_xy_multinomial(x, y)
        k = int(classes.shape[0])
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
            acceleration=self.acceleration,
            fit_intercept=self.fit_intercept,
            standardize_x=self.standardize,
            max_outer=self.max_outer,
            outer_tol=self.outer_tol,
        )
        coefs, intercepts, _, info = _multinomial_dispatch(
            payload,
            codes,
            k,
            kwargs,
            _core.solve_multinomial_elastic_net_path,
            _core.solve_multinomial_elastic_net_path_sparse,
        )
        coefs_3d = _bvec_to_coefs(coefs, p, k)
        self.coef_, self.intercept_ = _take_first(coefs_3d, intercepts)
        self.classes_ = classes
        self.info_ = info
        self.n_features_in_ = p
        self.n_classes_ = k
        return self


# =========================================================================
# Path classifiers
# =========================================================================


class _MultinomialPathBase(BaseEstimator):
    """Holds the path-fit results (`coefs_`, `intercepts_`, `lambdas_`,
    `classes_`, `info_`). Subclasses provide `_dense_fn`, `_sparse_fn`,
    and `_extra_kwargs()` for penalty-specific parameters (γ, a, α)."""

    info_: dict[str, Any]
    classes_: NDArray
    coefs_: NDArray[np.float64]
    intercepts_: NDArray[np.float64]
    lambdas_: NDArray[np.float64]
    n_features_in_: int
    n_classes_: int

    def _extra_kwargs(self) -> dict[str, Any]:
        return {}

    _dense_fn: Any
    _sparse_fn: Any

    def fit(self, x, y) -> "_MultinomialPathBase":
        payload, _n, p, codes, classes = _validate_xy_multinomial(x, y)
        k = int(classes.shape[0])
        lams = (
            np.ascontiguousarray(self.lambdas, dtype=np.float64)  # type: ignore[attr-defined]
            if self.lambdas is not None  # type: ignore[attr-defined]
            else None
        )
        w = (
            np.ascontiguousarray(self.weights, dtype=np.float64)  # type: ignore[attr-defined]
            if self.weights is not None  # type: ignore[attr-defined]
            else None
        )
        kwargs = dict(
            lambdas=lams,
            n_lambdas=self.n_lambdas,  # type: ignore[attr-defined]
            lambda_min_ratio=self.lambda_min_ratio,  # type: ignore[attr-defined]
            weights=w,
            max_iter=self.max_iter,  # type: ignore[attr-defined]
            tol=self.tol,  # type: ignore[attr-defined]
            acceleration=self.acceleration,  # type: ignore[attr-defined]
            fit_intercept=self.fit_intercept,  # type: ignore[attr-defined]
            standardize_x=self.standardize,  # type: ignore[attr-defined]
            max_outer=self.max_outer,  # type: ignore[attr-defined]
            outer_tol=self.outer_tol,  # type: ignore[attr-defined]
        )
        kwargs.update(self._extra_kwargs())
        coefs, intercepts, lambdas_used, info = _multinomial_dispatch(
            payload,
            codes,
            k,
            kwargs,
            self._dense_fn,
            self._sparse_fn,
        )
        self.coefs_ = _bvec_to_coefs(coefs, p, k)
        self.intercepts_ = intercepts.copy()
        self.lambdas_ = lambdas_used
        self.classes_ = classes
        self.info_ = info
        self.n_features_in_ = p
        self.n_classes_ = k
        return self


def _make_path_classifier(name, dense_fn, sparse_fn, extra_params):
    """Build a path-classifier class with the right __init__ surface and
    the path classifier `predict` family of methods (operating on the
    full-data refit at `lambdas_[best]`, but path estimators don't pick
    a single λ — they expose the whole path; per-λ inference is via
    `coefs_[idx]` directly).
    """
    raise NotImplementedError  # placeholder — keep concrete classes below


class MultinomialLassoPathClassifier(_MultinomialPathBase):
    """Path of multinomial-lasso fits along a λ-grid with warm starts.
    `coefs_` shape `(n_lambdas, K, p)`, `intercepts_` shape `(n_lambdas, K)`,
    `lambdas_` shape `(n_lambdas,)`. No single-best-λ refit — use
    `MultinomialLassoPathCV` for that."""

    _dense_fn = staticmethod(_core.solve_multinomial_lasso_path)
    _sparse_fn = staticmethod(_core.solve_multinomial_lasso_path_sparse)

    def __init__(
        self,
        *,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        max_outer: int = 25,
        outer_tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
    ) -> None:
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


class MultinomialMCPPathClassifier(_MultinomialPathBase):
    """Path of multinomial-MCP fits via LLA outer loop at each λ."""

    _dense_fn = staticmethod(_core.solve_multinomial_mcp_path)
    _sparse_fn = staticmethod(_core.solve_multinomial_mcp_path_sparse)

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
        max_outer: int = 25,
        outer_tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
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
        self.acceleration = acceleration

    def _extra_kwargs(self) -> dict[str, Any]:
        return {"gamma": self.gamma}


class MultinomialSCADPathClassifier(_MultinomialPathBase):
    """Path of multinomial-SCAD fits via LLA outer loop at each λ."""

    _dense_fn = staticmethod(_core.solve_multinomial_scad_path)
    _sparse_fn = staticmethod(_core.solve_multinomial_scad_path_sparse)

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
        max_outer: int = 25,
        outer_tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
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
        self.acceleration = acceleration

    def _extra_kwargs(self) -> dict[str, Any]:
        return {"a": self.a}


class MultinomialElasticNetPathClassifier(_MultinomialPathBase):
    """Path of multinomial elastic-net fits with warm starts."""

    _dense_fn = staticmethod(_core.solve_multinomial_elastic_net_path)
    _sparse_fn = staticmethod(_core.solve_multinomial_elastic_net_path_sparse)

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
        max_outer: int = 25,
        outer_tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
    ) -> None:
        self.alpha = alpha
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

    def _extra_kwargs(self) -> dict[str, Any]:
        return {"alpha": self.alpha}


# =========================================================================
# Path-CV classifiers
# =========================================================================


def _multinomial_deviance(
    y_true_codes: NDArray[np.float64],
    eta: NDArray[np.float64],
) -> float:
    """Mean per-sample multinomial deviance (negative log-likelihood):
    `(1/n) Σ_i (logsumexp(η_i) − η_{i, y_i})`. Lower-is-better."""
    n = y_true_codes.shape[0]
    m = eta.max(axis=1, keepdims=True)
    lse = np.log(np.exp(eta - m).sum(axis=1, keepdims=True)) + m
    lse = lse.ravel()
    rows = np.arange(n)
    eta_y = eta[rows, y_true_codes.astype(np.intp)]
    return float(np.mean(lse - eta_y))


class _MultinomialPathCVBase(BaseEstimator, ClassifierMixin, _MultinomialPredictMixin):
    """K-fold CV scaffold for multinomial classifiers. Default splitter
    is `StratifiedKFold` so class imbalance doesn't produce class-empty
    train folds. Score is multinomial deviance (lower-is-better)."""

    info_: dict[str, Any]
    classes_: NDArray
    coef_: NDArray[np.float64]
    intercept_: NDArray[np.float64]
    cv_scores_: NDArray[np.float64]
    cv_mean_scores_: NDArray[np.float64]
    cv_std_scores_: NDArray[np.float64]
    lambdas_: NDArray[np.float64]
    lambda_best_: float
    n_features_in_: int
    n_classes_: int

    def _make_base_path(self, **overrides):  # pragma: no cover - abstract
        raise NotImplementedError

    def fit(self, x, y) -> "_MultinomialPathCVBase":
        from sklearn.model_selection import StratifiedKFold

        _payload, n, p, codes, classes = _validate_xy_multinomial(x, y)
        k = int(classes.shape[0])

        if _is_sparse(x):
            from scipy import sparse  # type: ignore[import-untyped]

            x_for_indexing = x.tocsr() if not sparse.isspmatrix_csr(x) else x
        else:
            x_for_indexing = np.ascontiguousarray(x, dtype=np.float64)

        # Full-data fit gives the auto λ-grid AND the final refit.
        full = self._make_base_path().fit(x, y)
        lambdas = full.lambdas_

        cv = self.cv  # type: ignore[attr-defined]
        random_state = self.random_state  # type: ignore[attr-defined]
        if isinstance(cv, int):
            splitter = StratifiedKFold(
                n_splits=cv, shuffle=True, random_state=random_state
            )
        else:
            splitter = cv

        n_lambdas = lambdas.shape[0]
        # Stratified split needs the integer codes as the y for stratification.
        scores = np.full((splitter.get_n_splits(x_for_indexing, codes), n_lambdas), np.nan)

        for fold_idx, (train_idx, test_idx) in enumerate(
            splitter.split(x_for_indexing, codes)
        ):
            x_tr = x_for_indexing[train_idx]
            x_te = x_for_indexing[test_idx]
            y_tr = codes[train_idx]
            y_te = codes[test_idx]
            # Skip any fold accidentally missing a class — defensive.
            if np.unique(y_tr).shape[0] < k:
                continue
            fold = self._make_base_path(lambdas=lambdas).fit(x_tr, y_tr)
            for lam_idx in range(n_lambdas):
                eta = x_te @ fold.coefs_[lam_idx].T + fold.intercepts_[lam_idx]
                if hasattr(eta, "toarray"):
                    eta = eta.toarray()
                scores[fold_idx, lam_idx] = _multinomial_deviance(y_te, np.asarray(eta))

        self.cv_scores_ = scores
        self.cv_mean_scores_ = np.nanmean(scores, axis=0)
        self.cv_std_scores_ = np.nanstd(scores, axis=0)
        best = int(np.argmin(self.cv_mean_scores_))
        self.lambdas_ = lambdas
        self.lambda_best_ = float(lambdas[best])
        self.coef_ = full.coefs_[best]
        self.intercept_ = full.intercepts_[best].copy()
        self.classes_ = classes
        self.info_ = full.info_
        self.n_features_in_ = p
        self.n_classes_ = k
        return self


class MultinomialLassoPathCV(_MultinomialPathCVBase):
    """K-fold CV over a multinomial-lasso λ-path, scored by multinomial
    deviance."""

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
        max_outer: int = 25,
        outer_tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
    ) -> None:
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
        self.acceleration = acceleration

    def _make_base_path(self, **overrides) -> MultinomialLassoPathClassifier:
        kw: dict[str, Any] = dict(
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
            acceleration=self.acceleration,
        )
        kw.update(overrides)
        return MultinomialLassoPathClassifier(**kw)


class MultinomialMCPPathCV(_MultinomialPathCVBase):
    """K-fold CV over a multinomial-MCP λ-path."""

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
        max_outer: int = 25,
        outer_tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
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
        self.acceleration = acceleration

    def _make_base_path(self, **overrides) -> MultinomialMCPPathClassifier:
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
            acceleration=self.acceleration,
        )
        kw.update(overrides)
        return MultinomialMCPPathClassifier(**kw)


class MultinomialSCADPathCV(_MultinomialPathCVBase):
    """K-fold CV over a multinomial-SCAD λ-path."""

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
        max_outer: int = 25,
        outer_tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
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
        self.acceleration = acceleration

    def _make_base_path(self, **overrides) -> MultinomialSCADPathClassifier:
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
            acceleration=self.acceleration,
        )
        kw.update(overrides)
        return MultinomialSCADPathClassifier(**kw)


class MultinomialElasticNetPathCV(_MultinomialPathCVBase):
    """K-fold CV over a multinomial elastic-net λ-path."""

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
        max_outer: int = 25,
        outer_tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
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
        self.max_outer = max_outer
        self.outer_tol = outer_tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.acceleration = acceleration

    def _make_base_path(self, **overrides) -> MultinomialElasticNetPathClassifier:
        kw: dict[str, Any] = dict(
            alpha=self.alpha,
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
            acceleration=self.acceleration,
        )
        kw.update(overrides)
        return MultinomialElasticNetPathClassifier(**kw)


__all__ = [
    "MultinomialLassoClassifier",
    "MultinomialMCPClassifier",
    "MultinomialSCADClassifier",
    "MultinomialElasticNetClassifier",
    "MultinomialLassoPathClassifier",
    "MultinomialMCPPathClassifier",
    "MultinomialSCADPathClassifier",
    "MultinomialElasticNetPathClassifier",
    "MultinomialLassoPathCV",
    "MultinomialMCPPathCV",
    "MultinomialSCADPathCV",
    "MultinomialElasticNetPathCV",
]
