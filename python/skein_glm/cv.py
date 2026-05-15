"""Cross-validated λ selection for path estimators.

K-fold CV runs the underlying path estimator on each train fold over a
*shared* λ-grid, scores each held-out fold per λ, picks the λ that
minimizes the mean test score, and exposes the corresponding β as
`coef_` / `intercept_`.

Sequential folds for v0.1 — fold-level parallelism via Rayon is a
follow-up. Sparse input flows through the underlying path estimator
unchanged. The shared λ-grid comes from a single full-data fit (whose
coefs double as the final "refit", so there is no extra full-data
solve per fold).
"""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold

from skein_glm.estimators import (
    CoxGroupLassoPathRegressor,
    CoxGroupMCPPathRegressor,
    CoxMCPPathRegressor,
    CoxSCADPathRegressor,
    CoxSparseGroupLassoPathRegressor,
    CoxSparseGroupMCPPathRegressor,
    CoxSparseGroupSCADPathRegressor,
    BridgePathRegressor,
    ElasticNetPathRegressor,
    GroupElasticNetPathRegressor,
    GroupLassoPathRegressor,
    GroupMCPPathRegressor,
    GroupSCADPathRegressor,
    LogisticGroupLassoPathRegressor,
    LogisticGroupMCPPathRegressor,
    LogisticElasticNetPathRegressor,
    LogisticLassoPathRegressor,
    LogisticMCPPathRegressor,
    LogisticSCADPathRegressor,
    LogisticSparseGroupLassoPathRegressor,
    LogisticSparseGroupMCPPathRegressor,
    LogisticSparseGroupSCADPathRegressor,
    MCPPathRegressor,
    PoissonGroupLassoPathRegressor,
    PoissonGroupMCPPathRegressor,
    PoissonElasticNetPathRegressor,
    PoissonLassoPathRegressor,
    PoissonMCPPathRegressor,
    PoissonSCADPathRegressor,
    PoissonSparseGroupLassoPathRegressor,
    PoissonSparseGroupMCPPathRegressor,
    PoissonSparseGroupSCADPathRegressor,
    SCADPathRegressor,
    SparseGroupLassoPathRegressor,
    SparseGroupMCPPathRegressor,
    SparseGroupSCADPathRegressor,
    _is_sparse,
)
from sklearn.base import ClassifierMixin


def _row_index(x, idx: NDArray[np.int64]):
    """Subset rows of `x` by integer index. scipy.sparse CSC is bad at
    row indexing, so we convert to CSR first when sparse — the
    underlying path estimator will reconvert to CSC inside `fit`."""
    if _is_sparse(x):
        from scipy import sparse  # type: ignore[import-untyped]
        if not sparse.isspmatrix_csr(x):
            x = x.tocsr()
        return x[idx]
    return x[idx]


def _mse_score(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    diff = y_true - y_pred
    return float(np.mean(diff * diff))


class _PathCVMixin:
    """Shared K-fold CV machinery for path estimators.

    Subclasses implement `_make_base_path(**overrides)` to construct a
    fresh underlying `*PathRegressor` instance with the current
    hyper-parameters, optionally overriding a subset of them. They may
    also override `_score` (default: MSE, lower-is-better) and set
    `_score_higher_better=True` for higher-is-better scorers (AUC, R²).
    """

    coef_: NDArray[np.float64]
    intercept_: float
    cv_scores_: NDArray[np.float64]       # (n_folds, n_lambdas)
    cv_mean_scores_: NDArray[np.float64]  # (n_lambdas,)
    cv_std_scores_: NDArray[np.float64]   # (n_lambdas,)
    lambdas_: NDArray[np.float64]
    lambda_best_: float
    n_features_in_: int

    _score_higher_better: bool = False

    def _make_base_path(self, **overrides) -> Any:
        raise NotImplementedError

    def _score(self, y_true, y_pred) -> float:
        return _mse_score(y_true, y_pred)

    def _predict_for_score(self, model, x) -> NDArray[np.float64]:
        """Per-λ predictions on the held-out fold, in the form `_score`
        expects. Default: the path estimator's `predict(x)` (LS:
        Xβ+α). Subclasses override for GLMs (logistic →
        `predict_proba`, Cox → `decision_function`)."""
        return np.asarray(model.predict(x))

    def _fit_one_fold(
        self,
        train_idx: NDArray[np.int64],
        test_idx: NDArray[np.int64],
        x: Any,
        y: NDArray[np.float64],
        lambdas: NDArray[np.float64],
        offset_arr: NDArray[np.float64] | None,
    ) -> NDArray[np.float64]:
        """Fit on the train rows of one fold, score on the held-out
        rows, return the per-λ score vector. Called concurrently from
        the threaded fold loop in :meth:`fit`."""
        x_tr = _row_index(x, train_idx)
        x_te = _row_index(x, test_idx)
        y_tr = y[train_idx]
        y_te = y[test_idx]
        fold_overrides: dict[str, Any] = {"lambdas": lambdas}
        if offset_arr is not None:
            fold_overrides["offset"] = offset_arr[train_idx]
        fold_model = self._make_base_path(**fold_overrides).fit(x_tr, y_tr)
        preds = self._predict_for_score(fold_model, x_te)  # (n_te, n_lambdas)
        return np.array(
            [self._score(y_te, preds[:, k]) for k in range(len(lambdas))],
            dtype=np.float64,
        )

    def fit(self, x, y):
        from joblib import Parallel, delayed

        y = np.ascontiguousarray(y, dtype=np.float64)

        # Determine the λ-grid once. If the user passed `lambdas`, use
        # that verbatim; otherwise let the auto-grid come out of a
        # single full-data fit, which doubles as the refit producing
        # the final β at the chosen λ.
        explicit_lambdas = getattr(self, "lambdas", None)
        if explicit_lambdas is not None:
            lambdas = np.ascontiguousarray(explicit_lambdas, dtype=np.float64)
            full_fit = self._make_base_path(lambdas=lambdas).fit(x, y)
        else:
            full_fit = self._make_base_path().fit(x, y)
            lambdas = np.ascontiguousarray(full_fit.lambdas_, dtype=np.float64)

        # Resolve the fold splitter. Accept an int (KFold with shuffle)
        # or any sklearn-style CV splitter.
        cv = self.cv
        if isinstance(cv, int):
            splitter = KFold(
                n_splits=cv, shuffle=True, random_state=self.random_state
            )
        else:
            splitter = cv

        n_samples = x.shape[0]
        idx = np.arange(n_samples)

        # Per-sample offset (Poisson rate models). If the CV estimator
        # carries one, slice it along train_idx for each fold so the
        # underlying path estimator sees the correct n-vector.
        offset_arr = getattr(self, "offset", None)
        if offset_arr is not None:
            offset_arr = np.ascontiguousarray(offset_arr, dtype=np.float64)

        # Thread-based fold parallelism. The Rust solver releases the
        # GIL during compute, so threads run concurrently without GIL
        # contention; thread-based (vs process-based) avoids pickling X
        # into K worker processes. `n_jobs` is opt-in via the subclass
        # __init__ — subclasses that don't expose it default to serial
        # via the `getattr` fallback.
        n_jobs = getattr(self, "n_jobs", None)
        splits = list(splitter.split(idx))
        rows = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(self._fit_one_fold)(
                train_idx, test_idx, x, y, lambdas, offset_arr,
            )
            for train_idx, test_idx in splits
        )
        scores = np.stack(rows, axis=0)

        mean_scores = scores.mean(axis=0)
        std_scores = scores.std(axis=0)
        best_k = int(
            np.argmax(mean_scores) if self._score_higher_better else np.argmin(mean_scores)
        )

        self.coef_ = full_fit.coefs_[best_k]
        self.intercept_ = float(full_fit.intercepts_[best_k])
        self.cv_scores_ = scores
        self.cv_mean_scores_ = mean_scores
        self.cv_std_scores_ = std_scores
        self.lambdas_ = lambdas
        self.lambda_best_ = float(lambdas[best_k])
        self.n_features_in_ = full_fit.n_features_in_
        return self

    def predict(self, x) -> NDArray[np.float64]:
        if _is_sparse(x):
            return np.asarray(x @ self.coef_).ravel() + self.intercept_
        x = np.ascontiguousarray(x, dtype=np.float64)
        return x @ self.coef_ + self.intercept_


class MCPPathCV(_PathCVMixin, BaseEstimator, RegressorMixin):
    """K-fold cross-validated MCP path. Picks the λ minimizing mean
    test MSE and exposes the corresponding β as `coef_` / `intercept_`.

    The underlying solver is `MCPPathRegressor`. All of its constructor
    knobs are forwarded; CV adds `cv` (int K or sklearn splitter) and
    `random_state` (passed to `KFold` shuffle when `cv` is an int)."""

    def __init__(
        self,
        gamma: float = 3.0,
        *,
        cv: Any = 5,
        random_state: int | None = None,
        n_jobs: int | None = None,
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
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
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

    def _make_base_path(self, **overrides) -> MCPPathRegressor:
        kw: dict[str, Any] = dict(
            gamma=self.gamma,
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
        )
        kw.update(overrides)
        return MCPPathRegressor(**kw)


class SCADPathCV(_PathCVMixin, BaseEstimator, RegressorMixin):
    """K-fold cross-validated SCAD path. Same shape as `MCPPathCV`,
    but the underlying solver is `SCADPathRegressor`."""

    def __init__(
        self,
        a: float = 3.7,
        *,
        cv: Any = 5,
        random_state: int | None = None,
        n_jobs: int | None = None,
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
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
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

    def _make_base_path(self, **overrides) -> SCADPathRegressor:
        kw: dict[str, Any] = dict(
            a=self.a,
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
        )
        kw.update(overrides)
        return SCADPathRegressor(**kw)


class ElasticNetPathCV(_PathCVMixin, BaseEstimator, RegressorMixin):
    """K-fold cross-validated elastic-net path. Same shape as
    :class:`MCPPathCV`, but the underlying solver is
    :class:`ElasticNetPathRegressor`. Picks the λ minimizing mean
    test MSE on the supplied folds."""

    def __init__(
        self,
        alpha: float = 0.5,
        *,
        cv: Any = 5,
        random_state: int | None = None,
        n_jobs: int | None = None,
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
        self.alpha = alpha
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
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

    def _make_base_path(self, **overrides) -> ElasticNetPathRegressor:
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
        )
        kw.update(overrides)
        return ElasticNetPathRegressor(**kw)


class BridgePathCV(_PathCVMixin, BaseEstimator, RegressorMixin):
    """K-fold cross-validated bridge (ℓ_q) path. Picks the λ minimizing
    mean test MSE on the supplied folds. The non-convex inner objective
    means the chosen λ is a local-min selection — initialization
    (warm-starting from large λ down) makes this stable in practice
    but not guaranteed to be the global optimum at any λ."""

    def __init__(
        self,
        q: float = 0.5,
        *,
        eps: float = 1e-6,
        cv: Any = 5,
        random_state: int | None = None,
        n_jobs: int | None = None,
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
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
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

    def _make_base_path(self, **overrides) -> BridgePathRegressor:
        kw: dict[str, Any] = dict(
            q=self.q,
            eps=self.eps,
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
        return BridgePathRegressor(**kw)


# =====================================================================
# CV for LS group penalties (M5.1b)
# =====================================================================


class GroupLassoPathCV(_PathCVMixin, BaseEstimator, RegressorMixin):
    """K-fold CV over a group lasso λ-path. Picks the λ minimizing
    mean test MSE."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        *,
        cv: Any = 5,
        random_state: int | None = None,
        n_jobs: int | None = None,
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
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
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

    def _make_base_path(self, **overrides) -> GroupLassoPathRegressor:
        kw: dict[str, Any] = dict(
            groups=self.groups,
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
        return GroupLassoPathRegressor(**kw)


class GroupElasticNetPathCV(_PathCVMixin, BaseEstimator, RegressorMixin):
    """K-fold CV over a group elastic-net λ-path. Picks the λ
    minimizing mean test MSE."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        alpha: float = 0.5,
        *,
        cv: Any = 5,
        random_state: int | None = None,
        n_jobs: int | None = None,
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
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
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

    def _make_base_path(self, **overrides) -> GroupElasticNetPathRegressor:
        kw: dict[str, Any] = dict(
            groups=self.groups,
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
        return GroupElasticNetPathRegressor(**kw)


class GroupMCPPathCV(_PathCVMixin, BaseEstimator, RegressorMixin):
    """K-fold CV over a group MCP λ-path. The inner path solver is
    the native group-MCP block-CD shipped in v0.8 (no LLA outer
    loop) — see :class:`skein_glm.GroupMCPPathRegressor` for the
    algorithmic details and the ``max_outer`` / ``outer_tol``
    backward-compat notes."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        gamma: float = 3.0,
        *,
        cv: Any = 5,
        random_state: int | None = None,
        n_jobs: int | None = None,
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
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
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

    def _make_base_path(self, **overrides) -> GroupMCPPathRegressor:
        kw: dict[str, Any] = dict(
            groups=self.groups,
            gamma=self.gamma,
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
            max_outer=self.max_outer,
            outer_tol=self.outer_tol,
        )
        kw.update(overrides)
        return GroupMCPPathRegressor(**kw)


class GroupSCADPathCV(_PathCVMixin, BaseEstimator, RegressorMixin):
    """K-fold CV over a group SCAD λ-path (LLA outer loop). SCAD shape
    `a > 2` (default 3.7)."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        a: float = 3.7,
        *,
        cv: Any = 5,
        random_state: int | None = None,
        n_jobs: int | None = None,
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
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
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

    def _make_base_path(self, **overrides) -> GroupSCADPathRegressor:
        kw: dict[str, Any] = dict(
            groups=self.groups,
            a=self.a,
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
            max_outer=self.max_outer,
            outer_tol=self.outer_tol,
        )
        kw.update(overrides)
        return GroupSCADPathRegressor(**kw)


class SparseGroupLassoPathCV(_PathCVMixin, BaseEstimator, RegressorMixin):
    """K-fold CV over a sparse-group lasso λ-path."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        alpha: float = 0.5,
        *,
        cv: Any = 5,
        random_state: int | None = None,
        n_jobs: int | None = None,
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
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
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

    def _make_base_path(self, **overrides) -> SparseGroupLassoPathRegressor:
        kw: dict[str, Any] = dict(
            groups=self.groups,
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
        return SparseGroupLassoPathRegressor(**kw)


class SparseGroupMCPPathCV(_PathCVMixin, BaseEstimator, RegressorMixin):
    """K-fold CV over a sparse-group MCP λ-path (LLA outer loop)."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        gamma: float = 3.0,
        alpha: float = 0.5,
        *,
        cv: Any = 5,
        random_state: int | None = None,
        n_jobs: int | None = None,
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
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
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

    def _make_base_path(self, **overrides) -> SparseGroupMCPPathRegressor:
        kw: dict[str, Any] = dict(
            groups=self.groups,
            gamma=self.gamma,
            alpha=self.alpha,
            lambdas=self.lambdas,
            n_lambdas=self.n_lambdas,
            lambda_min_ratio=self.lambda_min_ratio,
            weights=self.weights,
            coord_weights=self.coord_weights,
            max_iter=self.max_iter,
            tol=self.tol,
            fit_intercept=self.fit_intercept,
            standardize=self.standardize,
            screening=self.screening,
            acceleration=self.acceleration,
            parallel=self.parallel,
            max_outer=self.max_outer,
            outer_tol=self.outer_tol,
        )
        kw.update(overrides)
        return SparseGroupMCPPathRegressor(**kw)


class SparseGroupSCADPathCV(_PathCVMixin, BaseEstimator, RegressorMixin):
    """K-fold CV over a sparse-group SCAD λ-path (LLA outer loop).
    SCAD shape `a > 2` (default 3.7)."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        a: float = 3.7,
        alpha: float = 0.5,
        *,
        cv: Any = 5,
        random_state: int | None = None,
        n_jobs: int | None = None,
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
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
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

    def _make_base_path(self, **overrides) -> SparseGroupSCADPathRegressor:
        kw: dict[str, Any] = dict(
            groups=self.groups,
            a=self.a,
            alpha=self.alpha,
            lambdas=self.lambdas,
            n_lambdas=self.n_lambdas,
            lambda_min_ratio=self.lambda_min_ratio,
            weights=self.weights,
            coord_weights=self.coord_weights,
            max_iter=self.max_iter,
            tol=self.tol,
            fit_intercept=self.fit_intercept,
            standardize=self.standardize,
            screening=self.screening,
            acceleration=self.acceleration,
            parallel=self.parallel,
            max_outer=self.max_outer,
            outer_tol=self.outer_tol,
        )
        kw.update(overrides)
        return SparseGroupSCADPathRegressor(**kw)


# =====================================================================
# GLM scorers and mixins (M5.1c)
# =====================================================================
#
# - Logistic: binomial deviance (lower-is-better). The CV's
#   `_predict_for_score` returns `predict_proba(x)` so the scorer sees
#   probabilities directly.
# - Poisson: Poisson deviance (lower-is-better). The path estimator's
#   `predict(x)` already returns `μ = exp(η)`, which matches what the
#   scorer expects.
# - Cox: Harrell's concordance index (higher-is-better). Cox CV gets
#   its own mixin because the fit signature is `fit(x, time, event)`
#   instead of `fit(x, y)`, the underlying solver has no intercept,
#   and the per-λ scoring uses the linear predictor η (not a
#   probability or rate).


def _sigmoid(z: NDArray[np.float64]) -> NDArray[np.float64]:
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def _logistic_deviance(y_true: NDArray[np.float64], p_pred: NDArray[np.float64]) -> float:
    p = np.clip(p_pred, 1e-15, 1.0 - 1e-15)
    return -2.0 * float(np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


def _poisson_deviance(y_true: NDArray[np.float64], mu_pred: NDArray[np.float64]) -> float:
    mu = np.maximum(mu_pred, 1e-12)
    # `y log(y/μ)` defined as 0 at y = 0 by convention; mask zeros
    # before the log to avoid `0 * -inf` warnings (numpy `where` is
    # not short-circuiting on the discarded branch).
    y_safe = np.where(y_true > 0, y_true, 1.0)
    y_log_y = np.where(y_true > 0, y_true * np.log(y_safe / mu), 0.0)
    return 2.0 * float(np.mean(y_log_y - (y_true - mu)))


def _harrell_c_index(
    time: NDArray[np.float64],
    event: NDArray[np.float64],
    eta: NDArray[np.float64],
) -> float:
    """Harrell's concordance index: fraction of comparable (i, j)
    pairs (where i had an earlier event and j was still at risk at
    `time[i]`) for which `η_i > η_j`. Ties in η count as 0.5. Returns
    0.5 when there are no comparable pairs (test fold with zero events
    or all events tied)."""
    n = len(time)
    num = 0.0
    den = 0.0
    for i in range(n):
        if event[i] < 0.5:
            continue
        for j in range(n):
            if time[j] <= time[i]:
                continue
            if eta[i] > eta[j]:
                num += 1.0
            elif eta[i] == eta[j]:
                num += 0.5
            den += 1.0
    if den < 0.5:
        return 0.5
    return num / den


class _LogisticPathCVMixin(_PathCVMixin):
    """CV for binary logistic path estimators. Scores by binomial
    deviance on `predict_proba`; final estimator exposes
    `decision_function`, `predict_proba`, and class-label `predict`."""

    _score_higher_better: bool = False

    def _predict_for_score(self, model, x) -> NDArray[np.float64]:
        return np.asarray(model.predict_proba(x))  # (n, n_lambdas)

    def _score(self, y_true, p_pred) -> float:
        return _logistic_deviance(y_true, p_pred)

    def decision_function(self, x) -> NDArray[np.float64]:
        # Linear predictor η = Xβ + α at the CV-selected λ.
        return _PathCVMixin.predict(self, x)

    def predict_proba(self, x) -> NDArray[np.float64]:
        return _sigmoid(self.decision_function(x))

    def predict(self, x) -> NDArray[np.float64]:
        return (self.predict_proba(x) >= 0.5).astype(np.float64)


class _PoissonPathCVMixin(_PathCVMixin):
    """CV for Poisson (log link) path estimators. Scores by Poisson
    deviance on `μ = exp(η)`; final estimator's `predict` returns the
    conditional mean, `decision_function` returns η."""

    _score_higher_better: bool = False

    def _predict_for_score(self, model, x) -> NDArray[np.float64]:
        # Path Poisson predict returns μ = exp(η) per λ.
        return np.asarray(model.predict(x))

    def _score(self, y_true, mu_pred) -> float:
        return _poisson_deviance(y_true, mu_pred)

    def decision_function(self, x) -> NDArray[np.float64]:
        return _PathCVMixin.predict(self, x)

    def predict(self, x) -> NDArray[np.float64]:
        return np.exp(self.decision_function(x))


class _CoxPathCVMixin:
    """CV for Cox PH path estimators. Different from `_PathCVMixin`:
    fit signature is `fit(x, time, event)`, no intercept, scoring uses
    Harrell's c-index on the linear predictor η. When `cv` is an int,
    folds are stratified by event indicator (preserves event count
    across folds — important when censoring is heavy). When `cv` is a
    pre-built splitter, it's used as-is."""

    coef_: NDArray[np.float64]
    cv_scores_: NDArray[np.float64]
    cv_mean_scores_: NDArray[np.float64]
    cv_std_scores_: NDArray[np.float64]
    lambdas_: NDArray[np.float64]
    lambda_best_: float
    n_features_in_: int

    _score_higher_better: bool = True

    def _make_base_path(self, **overrides) -> Any:
        raise NotImplementedError

    def _score(self, time_test, event_test, eta_pred) -> float:
        return _harrell_c_index(time_test, event_test, eta_pred)

    def _fit_one_fold_cox(
        self,
        train_idx: NDArray[np.int64],
        test_idx: NDArray[np.int64],
        x: Any,
        time_arr: NDArray[np.float64],
        event_arr: NDArray[np.float64],
        lambdas: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Fit one Cox fold; return per-λ c-index scores (NaN for the
        whole row if the fold has no events — those are masked out by
        nanmean / nanstd downstream)."""
        n_lambdas = len(lambdas)
        e_tr = event_arr[train_idx]
        if e_tr.sum() < 1.0:
            return np.full(n_lambdas, np.nan, dtype=np.float64)
        x_tr = _row_index(x, train_idx)
        x_te = _row_index(x, test_idx)
        t_tr = time_arr[train_idx]
        t_te = time_arr[test_idx]
        e_te = event_arr[test_idx]
        fold_model = self._make_base_path(lambdas=lambdas).fit(x_tr, t_tr, e_tr)
        etas = np.asarray(fold_model.decision_function(x_te))  # (n_te, n_lambdas)
        return np.array(
            [self._score(t_te, e_te, etas[:, k]) for k in range(n_lambdas)],
            dtype=np.float64,
        )

    def fit(self, x, time, event):
        from joblib import Parallel, delayed

        time_arr = np.ascontiguousarray(time, dtype=np.float64)
        event_arr = np.ascontiguousarray(event, dtype=np.float64)

        explicit_lambdas = getattr(self, "lambdas", None)
        if explicit_lambdas is not None:
            lambdas = np.ascontiguousarray(explicit_lambdas, dtype=np.float64)
            full_fit = self._make_base_path(lambdas=lambdas).fit(x, time_arr, event_arr)
        else:
            full_fit = self._make_base_path().fit(x, time_arr, event_arr)
            lambdas = np.ascontiguousarray(full_fit.lambdas_, dtype=np.float64)

        cv = self.cv
        if isinstance(cv, int):
            from sklearn.model_selection import StratifiedKFold
            splitter = StratifiedKFold(
                n_splits=cv, shuffle=True, random_state=self.random_state
            )
            stratify = event_arr.astype(int)
        else:
            splitter = cv
            stratify = None

        n_samples = x.shape[0]
        idx = np.arange(n_samples)
        split_iter = (
            splitter.split(idx, stratify) if stratify is not None
            else splitter.split(idx)
        )

        # Threaded fold parallelism — same rationale as the LS/GLM
        # mixin. `n_jobs` is opt-in via the subclass __init__.
        n_jobs = getattr(self, "n_jobs", None)
        splits = list(split_iter)
        rows = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(self._fit_one_fold_cox)(
                train_idx, test_idx, x, time_arr, event_arr, lambdas,
            )
            for train_idx, test_idx in splits
        )
        scores = np.stack(rows, axis=0)

        mean_scores = np.nanmean(scores, axis=0)
        std_scores = np.nanstd(scores, axis=0)
        best_k = int(
            np.argmax(mean_scores) if self._score_higher_better else np.argmin(mean_scores)
        )

        self.coef_ = full_fit.coefs_[best_k]
        self.cv_scores_ = scores
        self.cv_mean_scores_ = mean_scores
        self.cv_std_scores_ = std_scores
        self.lambdas_ = lambdas
        self.lambda_best_ = float(lambdas[best_k])
        self.n_features_in_ = full_fit.n_features_in_
        return self

    def decision_function(self, x) -> NDArray[np.float64]:
        if _is_sparse(x):
            return np.asarray(x @ self.coef_).ravel()
        x = np.ascontiguousarray(x, dtype=np.float64)
        return x @ self.coef_

    def predict(self, x) -> NDArray[np.float64]:
        # Cox prognostic index = linear predictor (matches the base
        # `CoxRegressor` convention).
        return self.decision_function(x)


# ---- Logistic CV wrappers (6) ------------------------------------------


class LogisticMCPPathCV(_LogisticPathCVMixin, BaseEstimator, ClassifierMixin):
    """K-fold CV over a logistic-MCP path. Picks the λ minimizing
    mean test binomial deviance."""

    def __init__(
        self,
        gamma: float = 3.0,
        *,
        cv: Any = 5,
        random_state: int | None = None,
        n_jobs: int | None = None,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.gamma = gamma
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def _make_base_path(self, **overrides) -> LogisticMCPPathRegressor:
        kw: dict[str, Any] = dict(
            gamma=self.gamma, lambdas=self.lambdas, n_lambdas=self.n_lambdas,
            lambda_min_ratio=self.lambda_min_ratio, weights=self.weights,
            max_iter=self.max_iter, tol=self.tol, fit_intercept=self.fit_intercept,
            acceleration=self.acceleration, max_outer=self.max_outer,
            outer_tol=self.outer_tol,
        )
        kw.update(overrides)
        return LogisticMCPPathRegressor(**kw)


class LogisticSCADPathCV(_LogisticPathCVMixin, BaseEstimator, ClassifierMixin):
    """K-fold CV over a logistic-SCAD path."""

    def __init__(
        self,
        a: float = 3.7,
        *,
        cv: Any = 5,
        random_state: int | None = None,
        n_jobs: int | None = None,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.a = a
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def _make_base_path(self, **overrides) -> LogisticSCADPathRegressor:
        kw: dict[str, Any] = dict(
            a=self.a, lambdas=self.lambdas, n_lambdas=self.n_lambdas,
            lambda_min_ratio=self.lambda_min_ratio, weights=self.weights,
            max_iter=self.max_iter, tol=self.tol, fit_intercept=self.fit_intercept,
            acceleration=self.acceleration, max_outer=self.max_outer,
            outer_tol=self.outer_tol,
        )
        kw.update(overrides)
        return LogisticSCADPathRegressor(**kw)


class LogisticElasticNetPathCV(_LogisticPathCVMixin, BaseEstimator, ClassifierMixin):
    """K-fold CV over a logistic + elastic-net path."""

    def __init__(
        self,
        alpha: float = 0.5,
        *,
        cv: Any = 5,
        random_state: int | None = None,
        n_jobs: int | None = None,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.alpha = alpha
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def _make_base_path(self, **overrides) -> LogisticElasticNetPathRegressor:
        kw: dict[str, Any] = dict(
            alpha=self.alpha, lambdas=self.lambdas, n_lambdas=self.n_lambdas,
            lambda_min_ratio=self.lambda_min_ratio, weights=self.weights,
            max_iter=self.max_iter, tol=self.tol, fit_intercept=self.fit_intercept,
            acceleration=self.acceleration, max_outer=self.max_outer,
            outer_tol=self.outer_tol,
        )
        kw.update(overrides)
        return LogisticElasticNetPathRegressor(**kw)


class LogisticLassoPathCV(_LogisticPathCVMixin, BaseEstimator, ClassifierMixin):
    """K-fold CV over a logistic-lasso path (proper convex L1, not the
    MCP-at-large-γ approximation)."""

    def __init__(
        self,
        *,
        cv: Any = 5,
        random_state: int | None = None,
        n_jobs: int | None = None,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def _make_base_path(self, **overrides) -> LogisticLassoPathRegressor:
        kw: dict[str, Any] = dict(
            lambdas=self.lambdas, n_lambdas=self.n_lambdas,
            lambda_min_ratio=self.lambda_min_ratio, weights=self.weights,
            max_iter=self.max_iter, tol=self.tol, fit_intercept=self.fit_intercept,
            acceleration=self.acceleration, max_outer=self.max_outer,
            outer_tol=self.outer_tol,
        )
        kw.update(overrides)
        return LogisticLassoPathRegressor(**kw)


class LogisticGroupLassoPathCV(_LogisticPathCVMixin, BaseEstimator, ClassifierMixin):
    """K-fold CV over a logistic + group-lasso path."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        *,
        cv: Any = 5,
        random_state: int | None = None,
        n_jobs: int | None = None,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.groups = groups
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def _make_base_path(self, **overrides) -> LogisticGroupLassoPathRegressor:
        kw: dict[str, Any] = dict(
            groups=self.groups, lambdas=self.lambdas, n_lambdas=self.n_lambdas,
            lambda_min_ratio=self.lambda_min_ratio, weights=self.weights,
            max_iter=self.max_iter, tol=self.tol, fit_intercept=self.fit_intercept,
            acceleration=self.acceleration, max_outer=self.max_outer,
            outer_tol=self.outer_tol,
        )
        kw.update(overrides)
        return LogisticGroupLassoPathRegressor(**kw)


class LogisticGroupMCPPathCV(_LogisticPathCVMixin, BaseEstimator, ClassifierMixin):
    """K-fold CV over a logistic + group-MCP path (LLA outer loop)."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        gamma: float = 3.0,
        *,
        cv: Any = 5,
        random_state: int | None = None,
        n_jobs: int | None = None,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.groups = groups
        self.gamma = gamma
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def _make_base_path(self, **overrides) -> LogisticGroupMCPPathRegressor:
        kw: dict[str, Any] = dict(
            groups=self.groups, gamma=self.gamma, lambdas=self.lambdas,
            n_lambdas=self.n_lambdas, lambda_min_ratio=self.lambda_min_ratio,
            weights=self.weights, max_iter=self.max_iter, tol=self.tol,
            fit_intercept=self.fit_intercept, acceleration=self.acceleration,
            max_outer=self.max_outer, outer_tol=self.outer_tol,
        )
        kw.update(overrides)
        return LogisticGroupMCPPathRegressor(**kw)


class LogisticSparseGroupLassoPathCV(_LogisticPathCVMixin, BaseEstimator, ClassifierMixin):
    """K-fold CV over a logistic + sparse-group-lasso path."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        alpha: float = 0.5,
        *,
        cv: Any = 5,
        random_state: int | None = None,
        n_jobs: int | None = None,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.groups = groups
        self.alpha = alpha
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def _make_base_path(self, **overrides) -> LogisticSparseGroupLassoPathRegressor:
        kw: dict[str, Any] = dict(
            groups=self.groups, alpha=self.alpha, lambdas=self.lambdas,
            n_lambdas=self.n_lambdas, lambda_min_ratio=self.lambda_min_ratio,
            weights=self.weights, max_iter=self.max_iter, tol=self.tol,
            fit_intercept=self.fit_intercept, acceleration=self.acceleration,
            max_outer=self.max_outer, outer_tol=self.outer_tol,
        )
        kw.update(overrides)
        return LogisticSparseGroupLassoPathRegressor(**kw)


class LogisticSparseGroupMCPPathCV(_LogisticPathCVMixin, BaseEstimator, ClassifierMixin):
    """K-fold CV over a logistic + sparse-group-MCP path."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        gamma: float = 3.0,
        alpha: float = 0.5,
        *,
        cv: Any = 5,
        random_state: int | None = None,
        n_jobs: int | None = None,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        weights: NDArray[np.float64] | None = None,
        coord_weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.groups = groups
        self.gamma = gamma
        self.alpha = alpha
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.weights = weights
        self.coord_weights = coord_weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def _make_base_path(self, **overrides) -> LogisticSparseGroupMCPPathRegressor:
        kw: dict[str, Any] = dict(
            groups=self.groups, gamma=self.gamma, alpha=self.alpha,
            lambdas=self.lambdas, n_lambdas=self.n_lambdas,
            lambda_min_ratio=self.lambda_min_ratio, weights=self.weights,
            coord_weights=self.coord_weights,
            max_iter=self.max_iter, tol=self.tol,
            fit_intercept=self.fit_intercept, acceleration=self.acceleration,
            max_outer=self.max_outer, outer_tol=self.outer_tol,
        )
        kw.update(overrides)
        return LogisticSparseGroupMCPPathRegressor(**kw)


class LogisticSparseGroupSCADPathCV(_LogisticPathCVMixin, BaseEstimator, ClassifierMixin):
    """K-fold CV over a logistic + sparse-group-SCAD path."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        a: float = 3.7,
        alpha: float = 0.5,
        *,
        cv: Any = 5,
        random_state: int | None = None,
        n_jobs: int | None = None,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        weights: NDArray[np.float64] | None = None,
        coord_weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.groups = groups
        self.a = a
        self.alpha = alpha
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.weights = weights
        self.coord_weights = coord_weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def _make_base_path(self, **overrides) -> LogisticSparseGroupSCADPathRegressor:
        kw: dict[str, Any] = dict(
            groups=self.groups, a=self.a, alpha=self.alpha,
            lambdas=self.lambdas, n_lambdas=self.n_lambdas,
            lambda_min_ratio=self.lambda_min_ratio, weights=self.weights,
            coord_weights=self.coord_weights,
            max_iter=self.max_iter, tol=self.tol,
            fit_intercept=self.fit_intercept, acceleration=self.acceleration,
            max_outer=self.max_outer, outer_tol=self.outer_tol,
        )
        kw.update(overrides)
        return LogisticSparseGroupSCADPathRegressor(**kw)


# ---- Poisson CV wrappers (6) -------------------------------------------


class PoissonMCPPathCV(_PoissonPathCVMixin, BaseEstimator, RegressorMixin):
    """K-fold CV over a Poisson-MCP path. Picks λ minimizing mean
    test Poisson deviance."""

    def __init__(
        self,
        gamma: float = 3.0,
        *,
        cv: Any = 5,
        random_state: int | None = None,
        n_jobs: int | None = None,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        offset: NDArray[np.float64] | None = None,
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.gamma = gamma
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.offset = offset
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def _make_base_path(self, **overrides) -> PoissonMCPPathRegressor:
        kw: dict[str, Any] = dict(
            gamma=self.gamma, lambdas=self.lambdas, n_lambdas=self.n_lambdas,
            lambda_min_ratio=self.lambda_min_ratio, offset=self.offset, weights=self.weights,
            max_iter=self.max_iter, tol=self.tol, fit_intercept=self.fit_intercept,
            acceleration=self.acceleration, max_outer=self.max_outer,
            outer_tol=self.outer_tol,
        )
        kw.update(overrides)
        return PoissonMCPPathRegressor(**kw)


class PoissonSCADPathCV(_PoissonPathCVMixin, BaseEstimator, RegressorMixin):
    """K-fold CV over a Poisson-SCAD path."""

    def __init__(
        self,
        a: float = 3.7,
        *,
        cv: Any = 5,
        random_state: int | None = None,
        n_jobs: int | None = None,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        offset: NDArray[np.float64] | None = None,
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.a = a
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.offset = offset
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def _make_base_path(self, **overrides) -> PoissonSCADPathRegressor:
        kw: dict[str, Any] = dict(
            a=self.a, lambdas=self.lambdas, n_lambdas=self.n_lambdas,
            lambda_min_ratio=self.lambda_min_ratio, offset=self.offset, weights=self.weights,
            max_iter=self.max_iter, tol=self.tol, fit_intercept=self.fit_intercept,
            acceleration=self.acceleration, max_outer=self.max_outer,
            outer_tol=self.outer_tol,
        )
        kw.update(overrides)
        return PoissonSCADPathRegressor(**kw)


class PoissonElasticNetPathCV(_PoissonPathCVMixin, BaseEstimator, RegressorMixin):
    """K-fold CV over a Poisson + elastic-net path."""

    def __init__(
        self,
        alpha: float = 0.5,
        *,
        cv: Any = 5,
        random_state: int | None = None,
        n_jobs: int | None = None,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        offset: NDArray[np.float64] | None = None,
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.alpha = alpha
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.offset = offset
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def _make_base_path(self, **overrides) -> PoissonElasticNetPathRegressor:
        kw: dict[str, Any] = dict(
            alpha=self.alpha, lambdas=self.lambdas, n_lambdas=self.n_lambdas,
            lambda_min_ratio=self.lambda_min_ratio, offset=self.offset, weights=self.weights,
            max_iter=self.max_iter, tol=self.tol, fit_intercept=self.fit_intercept,
            acceleration=self.acceleration, max_outer=self.max_outer,
            outer_tol=self.outer_tol,
        )
        kw.update(overrides)
        return PoissonElasticNetPathRegressor(**kw)


class PoissonLassoPathCV(_PoissonPathCVMixin, BaseEstimator, RegressorMixin):
    """K-fold CV over a Poisson-lasso path (proper convex L1)."""

    def __init__(
        self,
        *,
        cv: Any = 5,
        random_state: int | None = None,
        n_jobs: int | None = None,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        offset: NDArray[np.float64] | None = None,
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.offset = offset
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def _make_base_path(self, **overrides) -> PoissonLassoPathRegressor:
        kw: dict[str, Any] = dict(
            lambdas=self.lambdas, n_lambdas=self.n_lambdas,
            lambda_min_ratio=self.lambda_min_ratio, offset=self.offset, weights=self.weights,
            max_iter=self.max_iter, tol=self.tol, fit_intercept=self.fit_intercept,
            acceleration=self.acceleration, max_outer=self.max_outer,
            outer_tol=self.outer_tol,
        )
        kw.update(overrides)
        return PoissonLassoPathRegressor(**kw)


class PoissonGroupLassoPathCV(_PoissonPathCVMixin, BaseEstimator, RegressorMixin):
    """K-fold CV over a Poisson + group-lasso path."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        *,
        cv: Any = 5,
        random_state: int | None = None,
        n_jobs: int | None = None,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        offset: NDArray[np.float64] | None = None,
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.groups = groups
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.offset = offset
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def _make_base_path(self, **overrides) -> PoissonGroupLassoPathRegressor:
        kw: dict[str, Any] = dict(
            groups=self.groups, lambdas=self.lambdas, n_lambdas=self.n_lambdas,
            lambda_min_ratio=self.lambda_min_ratio, offset=self.offset, weights=self.weights,
            max_iter=self.max_iter, tol=self.tol, fit_intercept=self.fit_intercept,
            acceleration=self.acceleration, max_outer=self.max_outer,
            outer_tol=self.outer_tol,
        )
        kw.update(overrides)
        return PoissonGroupLassoPathRegressor(**kw)


class PoissonGroupMCPPathCV(_PoissonPathCVMixin, BaseEstimator, RegressorMixin):
    """K-fold CV over a Poisson + group-MCP path."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        gamma: float = 3.0,
        *,
        cv: Any = 5,
        random_state: int | None = None,
        n_jobs: int | None = None,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        offset: NDArray[np.float64] | None = None,
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.groups = groups
        self.gamma = gamma
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.offset = offset
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def _make_base_path(self, **overrides) -> PoissonGroupMCPPathRegressor:
        kw: dict[str, Any] = dict(
            groups=self.groups, gamma=self.gamma, lambdas=self.lambdas,
            n_lambdas=self.n_lambdas, lambda_min_ratio=self.lambda_min_ratio,
            offset=self.offset, weights=self.weights, max_iter=self.max_iter, tol=self.tol,
            fit_intercept=self.fit_intercept, acceleration=self.acceleration,
            max_outer=self.max_outer, outer_tol=self.outer_tol,
        )
        kw.update(overrides)
        return PoissonGroupMCPPathRegressor(**kw)


class PoissonSparseGroupLassoPathCV(_PoissonPathCVMixin, BaseEstimator, RegressorMixin):
    """K-fold CV over a Poisson + sparse-group-lasso path."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        alpha: float = 0.5,
        *,
        cv: Any = 5,
        random_state: int | None = None,
        n_jobs: int | None = None,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        offset: NDArray[np.float64] | None = None,
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.groups = groups
        self.alpha = alpha
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.offset = offset
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def _make_base_path(self, **overrides) -> PoissonSparseGroupLassoPathRegressor:
        kw: dict[str, Any] = dict(
            groups=self.groups, alpha=self.alpha, lambdas=self.lambdas,
            n_lambdas=self.n_lambdas, lambda_min_ratio=self.lambda_min_ratio,
            offset=self.offset, weights=self.weights, max_iter=self.max_iter, tol=self.tol,
            fit_intercept=self.fit_intercept, acceleration=self.acceleration,
            max_outer=self.max_outer, outer_tol=self.outer_tol,
        )
        kw.update(overrides)
        return PoissonSparseGroupLassoPathRegressor(**kw)


class PoissonSparseGroupMCPPathCV(_PoissonPathCVMixin, BaseEstimator, RegressorMixin):
    """K-fold CV over a Poisson + sparse-group-MCP path."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        gamma: float = 3.0,
        alpha: float = 0.5,
        *,
        cv: Any = 5,
        random_state: int | None = None,
        n_jobs: int | None = None,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        offset: NDArray[np.float64] | None = None,
        weights: NDArray[np.float64] | None = None,
        coord_weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.groups = groups
        self.gamma = gamma
        self.alpha = alpha
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.offset = offset
        self.weights = weights
        self.coord_weights = coord_weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def _make_base_path(self, **overrides) -> PoissonSparseGroupMCPPathRegressor:
        kw: dict[str, Any] = dict(
            groups=self.groups, gamma=self.gamma, alpha=self.alpha,
            lambdas=self.lambdas, n_lambdas=self.n_lambdas,
            lambda_min_ratio=self.lambda_min_ratio, offset=self.offset, weights=self.weights,
            coord_weights=self.coord_weights,
            max_iter=self.max_iter, tol=self.tol,
            fit_intercept=self.fit_intercept, acceleration=self.acceleration,
            max_outer=self.max_outer, outer_tol=self.outer_tol,
        )
        kw.update(overrides)
        return PoissonSparseGroupMCPPathRegressor(**kw)


class PoissonSparseGroupSCADPathCV(_PoissonPathCVMixin, BaseEstimator, RegressorMixin):
    """K-fold CV over a Poisson + sparse-group-SCAD path."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        a: float = 3.7,
        alpha: float = 0.5,
        *,
        cv: Any = 5,
        random_state: int | None = None,
        n_jobs: int | None = None,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        offset: NDArray[np.float64] | None = None,
        weights: NDArray[np.float64] | None = None,
        coord_weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.groups = groups
        self.a = a
        self.alpha = alpha
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.offset = offset
        self.weights = weights
        self.coord_weights = coord_weights
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def _make_base_path(self, **overrides) -> PoissonSparseGroupSCADPathRegressor:
        kw: dict[str, Any] = dict(
            groups=self.groups, a=self.a, alpha=self.alpha,
            lambdas=self.lambdas, n_lambdas=self.n_lambdas,
            lambda_min_ratio=self.lambda_min_ratio, offset=self.offset, weights=self.weights,
            coord_weights=self.coord_weights,
            max_iter=self.max_iter, tol=self.tol,
            fit_intercept=self.fit_intercept, acceleration=self.acceleration,
            max_outer=self.max_outer, outer_tol=self.outer_tol,
        )
        kw.update(overrides)
        return PoissonSparseGroupSCADPathRegressor(**kw)


# ---- Cox CV wrappers (6) -----------------------------------------------


class CoxMCPPathCV(_CoxPathCVMixin, BaseEstimator):
    """K-fold CV over a Cox-MCP path. Picks λ maximizing mean test
    Harrell concordance index."""

    def __init__(
        self,
        gamma: float = 3.0,
        *,
        cv: Any = 5,
        random_state: int | None = None,
        n_jobs: int | None = None,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        ties: str = 'breslow',
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.gamma = gamma
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.ties = ties
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def _make_base_path(self, **overrides) -> CoxMCPPathRegressor:
        kw: dict[str, Any] = dict(
            gamma=self.gamma, lambdas=self.lambdas, n_lambdas=self.n_lambdas,
            lambda_min_ratio=self.lambda_min_ratio, ties=self.ties, weights=self.weights,
            max_iter=self.max_iter, tol=self.tol,
            acceleration=self.acceleration, max_outer=self.max_outer,
            outer_tol=self.outer_tol,
        )
        kw.update(overrides)
        return CoxMCPPathRegressor(**kw)


class CoxSCADPathCV(_CoxPathCVMixin, BaseEstimator):
    """K-fold CV over a Cox-SCAD path."""

    def __init__(
        self,
        a: float = 3.7,
        *,
        cv: Any = 5,
        random_state: int | None = None,
        n_jobs: int | None = None,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        ties: str = 'breslow',
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.a = a
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.ties = ties
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def _make_base_path(self, **overrides) -> CoxSCADPathRegressor:
        kw: dict[str, Any] = dict(
            a=self.a, lambdas=self.lambdas, n_lambdas=self.n_lambdas,
            lambda_min_ratio=self.lambda_min_ratio, ties=self.ties, weights=self.weights,
            max_iter=self.max_iter, tol=self.tol,
            acceleration=self.acceleration, max_outer=self.max_outer,
            outer_tol=self.outer_tol,
        )
        kw.update(overrides)
        return CoxSCADPathRegressor(**kw)


class CoxGroupLassoPathCV(_CoxPathCVMixin, BaseEstimator):
    """K-fold CV over a Cox + group-lasso path."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        *,
        cv: Any = 5,
        random_state: int | None = None,
        n_jobs: int | None = None,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        ties: str = 'breslow',
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.groups = groups
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.ties = ties
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def _make_base_path(self, **overrides) -> CoxGroupLassoPathRegressor:
        kw: dict[str, Any] = dict(
            groups=self.groups, lambdas=self.lambdas, n_lambdas=self.n_lambdas,
            lambda_min_ratio=self.lambda_min_ratio, ties=self.ties, weights=self.weights,
            max_iter=self.max_iter, tol=self.tol,
            acceleration=self.acceleration, max_outer=self.max_outer,
            outer_tol=self.outer_tol,
        )
        kw.update(overrides)
        return CoxGroupLassoPathRegressor(**kw)


class CoxGroupMCPPathCV(_CoxPathCVMixin, BaseEstimator):
    """K-fold CV over a Cox + group-MCP path."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        gamma: float = 3.0,
        *,
        cv: Any = 5,
        random_state: int | None = None,
        n_jobs: int | None = None,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        ties: str = 'breslow',
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.groups = groups
        self.gamma = gamma
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.ties = ties
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def _make_base_path(self, **overrides) -> CoxGroupMCPPathRegressor:
        kw: dict[str, Any] = dict(
            groups=self.groups, gamma=self.gamma, lambdas=self.lambdas,
            n_lambdas=self.n_lambdas, lambda_min_ratio=self.lambda_min_ratio,
            ties=self.ties, weights=self.weights, max_iter=self.max_iter, tol=self.tol,
            acceleration=self.acceleration, max_outer=self.max_outer,
            outer_tol=self.outer_tol,
        )
        kw.update(overrides)
        return CoxGroupMCPPathRegressor(**kw)


class CoxSparseGroupLassoPathCV(_CoxPathCVMixin, BaseEstimator):
    """K-fold CV over a Cox + sparse-group-lasso path."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        alpha: float = 0.5,
        *,
        cv: Any = 5,
        random_state: int | None = None,
        n_jobs: int | None = None,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        ties: str = 'breslow',
        weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.groups = groups
        self.alpha = alpha
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.ties = ties
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def _make_base_path(self, **overrides) -> CoxSparseGroupLassoPathRegressor:
        kw: dict[str, Any] = dict(
            groups=self.groups, alpha=self.alpha, lambdas=self.lambdas,
            n_lambdas=self.n_lambdas, lambda_min_ratio=self.lambda_min_ratio,
            ties=self.ties, weights=self.weights, max_iter=self.max_iter, tol=self.tol,
            acceleration=self.acceleration, max_outer=self.max_outer,
            outer_tol=self.outer_tol,
        )
        kw.update(overrides)
        return CoxSparseGroupLassoPathRegressor(**kw)


class CoxSparseGroupMCPPathCV(_CoxPathCVMixin, BaseEstimator):
    """K-fold CV over a Cox + sparse-group-MCP path."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        gamma: float = 3.0,
        alpha: float = 0.5,
        *,
        cv: Any = 5,
        random_state: int | None = None,
        n_jobs: int | None = None,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        ties: str = 'breslow',
        weights: NDArray[np.float64] | None = None,
        coord_weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.groups = groups
        self.gamma = gamma
        self.alpha = alpha
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.ties = ties
        self.weights = weights
        self.coord_weights = coord_weights
        self.max_iter = max_iter
        self.tol = tol
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def _make_base_path(self, **overrides) -> CoxSparseGroupMCPPathRegressor:
        kw: dict[str, Any] = dict(
            groups=self.groups, gamma=self.gamma, alpha=self.alpha,
            lambdas=self.lambdas, n_lambdas=self.n_lambdas,
            lambda_min_ratio=self.lambda_min_ratio, ties=self.ties, weights=self.weights,
            coord_weights=self.coord_weights,
            max_iter=self.max_iter, tol=self.tol,
            acceleration=self.acceleration, max_outer=self.max_outer,
            outer_tol=self.outer_tol,
        )
        kw.update(overrides)
        return CoxSparseGroupMCPPathRegressor(**kw)


class CoxSparseGroupSCADPathCV(_CoxPathCVMixin, BaseEstimator):
    """K-fold CV over a Cox + sparse-group-SCAD path."""

    def __init__(
        self,
        groups: NDArray[np.int64],
        a: float = 3.7,
        alpha: float = 0.5,
        *,
        cv: Any = 5,
        random_state: int | None = None,
        n_jobs: int | None = None,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        ties: str = 'breslow',
        weights: NDArray[np.float64] | None = None,
        coord_weights: NDArray[np.float64] | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        acceleration: int | None = 5,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
    ) -> None:
        self.groups = groups
        self.a = a
        self.alpha = alpha
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.ties = ties
        self.weights = weights
        self.coord_weights = coord_weights
        self.max_iter = max_iter
        self.tol = tol
        self.acceleration = acceleration
        self.max_outer = max_outer
        self.outer_tol = outer_tol

    def _make_base_path(self, **overrides) -> CoxSparseGroupSCADPathRegressor:
        kw: dict[str, Any] = dict(
            groups=self.groups, a=self.a, alpha=self.alpha,
            lambdas=self.lambdas, n_lambdas=self.n_lambdas,
            lambda_min_ratio=self.lambda_min_ratio, ties=self.ties, weights=self.weights,
            coord_weights=self.coord_weights,
            max_iter=self.max_iter, tol=self.tol,
            acceleration=self.acceleration, max_outer=self.max_outer,
            outer_tol=self.outer_tol,
        )
        kw.update(overrides)
        return CoxSparseGroupSCADPathRegressor(**kw)
