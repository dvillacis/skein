"""Adaptive {lasso, MCP, SCAD} estimators (M6.x).

Two-stage procedure introduced by Zou (2006) for adaptive lasso, extended
here to MCP and SCAD:

1. **Pilot fit.** Run a plain lasso path (MCP at γ → ∞) on `(X, y)` and
   pick β at a position `pilot_position` along the path (default
   `'mid'` — the middle of the auto-generated λ-grid). Other options:
   `'last'` (smallest λ, closest to OLS), or an integer index.
2. **Adaptive weights.** Compute per-feature `w_j = 1 / max(|β_pilot[j]|,
   eps_pilot)^η`. Larger pilot magnitudes ⇒ smaller penalty weight ⇒ less
   shrinkage on truly active features. The exponent `η` (default 1.0)
   controls how aggressively the weights pull toward unbiased estimates.
3. **Final fit.** Run the chosen final penalty (Lasso / MCP / SCAD) on
   `(X, y)` with these adaptive weights. The path is the **final**
   estimator's path, λ-decreasing.

The pilot fit is on the **full** data (not per-fold) so the CV variants
keep their statistical guarantees: the pilot weights are an
input-data-based summary, the per-fold fit only re-runs the final
adaptive solve on each train split.

This is the headline reason the per-feature weight axis exists in
skein — the underlying solvers all accept `weights=` already, so
adaptive estimators are pure composition with no Rust changes.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import KFold

from skein_glm.cv import (
    _CoxPathCVMixin,
    _LogisticPathCVMixin,
    _PoissonPathCVMixin,
)
from skein_glm.estimators import (
    CoxMCPPathRegressor,
    CoxSCADPathRegressor,
    GroupLassoPathRegressor,
    GroupMCPPathRegressor,
    GroupSCADPathRegressor,
    LogisticLassoPathRegressor,
    LogisticMCPPathRegressor,
    LogisticSCADPathRegressor,
    MCPPathRegressor,
    PoissonLassoPathRegressor,
    PoissonMCPPathRegressor,
    PoissonSCADPathRegressor,
    SCADPathRegressor,
    _is_sparse,
)

PilotPosition = Literal["mid", "last"]


def _validate_pilot_position(p: Any, n_lambdas: int) -> int:
    """Resolve `pilot_position` to a concrete index in `[0, n_lambdas)`."""
    if isinstance(p, str):
        if p == "mid":
            return n_lambdas // 2
        if p == "last":
            return n_lambdas - 1
        raise ValueError(
            f"pilot_position must be 'mid', 'last', or int; got {p!r}"
        )
    if isinstance(p, (int, np.integer)):
        idx = int(p)
        if not 0 <= idx < n_lambdas:
            raise ValueError(
                f"pilot_position {idx} out of range for n_pilot_lambdas={n_lambdas}"
            )
        return idx
    raise TypeError(
        f"pilot_position must be 'mid', 'last', or int; got {type(p).__name__}"
    )


def _adaptive_weights(
    beta_pilot: NDArray[np.float64],
    eta: float,
    eps_pilot: float,
) -> NDArray[np.float64]:
    """Per-feature `1 / max(|β|, eps)^η`. Used as the `weights=`
    argument to the final solver."""
    if not eta > 0.0:
        raise ValueError(f"eta must be > 0; got {eta}")
    if not eps_pilot > 0.0:
        raise ValueError(f"eps_pilot must be > 0; got {eps_pilot}")
    mag = np.maximum(np.abs(beta_pilot), eps_pilot)
    return 1.0 / np.power(mag, eta)


def _adaptive_weights_per_group(
    beta_pilot: NDArray[np.float64],
    groups: NDArray[np.int64],
    eta: float,
    eps_pilot: float,
) -> NDArray[np.float64]:
    """Per-group `w_g = 1 / max(‖β_pilot[g]‖_2, ε)^η`. Returns one weight
    per unique group label, in the same label order as the underlying
    Rust `Groups` (ascending integer labels starting at 0)."""
    if not eta > 0.0:
        raise ValueError(f"eta must be > 0; got {eta}")
    if not eps_pilot > 0.0:
        raise ValueError(f"eps_pilot must be > 0; got {eps_pilot}")
    groups_arr = np.asarray(groups, dtype=np.int64)
    if groups_arr.shape[0] != beta_pilot.shape[0]:
        raise ValueError(
            f"groups length {groups_arr.shape[0]} does not match β length {beta_pilot.shape[0]}"
        )
    n_groups = int(groups_arr.max()) + 1
    norms = np.zeros(n_groups, dtype=np.float64)
    for g in range(n_groups):
        mask = groups_arr == g
        norms[g] = float(np.linalg.norm(beta_pilot[mask]))
    mag = np.maximum(norms, eps_pilot)
    return 1.0 / np.power(mag, eta)


def _fit_pilot(
    x,
    y,
    n_pilot_lambdas: int,
    pilot_position,
    fit_intercept: bool,
    standardize: bool,
) -> NDArray[np.float64]:
    """Run a plain lasso path (MCP at γ = 1e9) on `(X, y)` and return the
    β at the resolved `pilot_position`. The path solver dispatches on
    sparse vs dense via the underlying estimator."""
    pilot = MCPPathRegressor(
        gamma=1e9,
        n_lambdas=n_pilot_lambdas,
        lambda_min_ratio=1e-3,
        fit_intercept=fit_intercept,
        standardize=standardize,
    ).fit(x, y)
    idx = _validate_pilot_position(pilot_position, len(pilot.lambdas_))
    return pilot.coefs_[idx].copy()


# =========================================================================
# Path estimators
# =========================================================================


class _AdaptivePathBase(BaseEstimator, RegressorMixin):
    """Common scaffold: pilot → weights → final path. Subclasses set
    `_final_cls` (the underlying skein path estimator class) and
    `_extra_kwargs()` (penalty-specific params like γ or a)."""

    coefs_: NDArray[np.float64]
    intercepts_: NDArray[np.float64]
    lambdas_: NDArray[np.float64]
    coef_pilot_: NDArray[np.float64]
    weights_: NDArray[np.float64]
    info_: dict[str, Any]
    n_features_in_: int

    _final_cls: Any  # subclass-defined

    def _extra_kwargs(self) -> dict[str, Any]:
        return {}

    def _make_final(self, weights: NDArray[np.float64]):
        kw: dict[str, Any] = dict(
            lambdas=self.lambdas,  # type: ignore[attr-defined]
            n_lambdas=self.n_lambdas,  # type: ignore[attr-defined]
            lambda_min_ratio=self.lambda_min_ratio,  # type: ignore[attr-defined]
            weights=weights,
            max_iter=self.max_iter,  # type: ignore[attr-defined]
            tol=self.tol,  # type: ignore[attr-defined]
            fit_intercept=self.fit_intercept,  # type: ignore[attr-defined]
            standardize=self.standardize,  # type: ignore[attr-defined]
            screening=self.screening,  # type: ignore[attr-defined]
            acceleration=self.acceleration,  # type: ignore[attr-defined]
        )
        kw.update(self._extra_kwargs())
        return self._final_cls(**kw)

    def fit(self, x, y) -> "_AdaptivePathBase":
        beta_pilot = _fit_pilot(
            x,
            y,
            self.n_pilot_lambdas,  # type: ignore[attr-defined]
            self.pilot_position,  # type: ignore[attr-defined]
            self.fit_intercept,  # type: ignore[attr-defined]
            self.standardize,  # type: ignore[attr-defined]
        )
        weights = _adaptive_weights(
            beta_pilot,
            self.eta,  # type: ignore[attr-defined]
            self.eps_pilot,  # type: ignore[attr-defined]
        )
        final = self._make_final(weights).fit(x, y)
        self.coefs_ = final.coefs_
        self.intercepts_ = final.intercepts_
        self.lambdas_ = final.lambdas_
        self.coef_pilot_ = beta_pilot
        self.weights_ = weights
        self.info_ = final.info_
        self.n_features_in_ = final.n_features_in_
        return self

    def predict(self, x) -> NDArray[np.float64]:
        if _is_sparse(x):
            return np.asarray(x @ self.coefs_.T) + self.intercepts_[None, :]
        x = np.ascontiguousarray(x, dtype=np.float64)
        return x @ self.coefs_.T + self.intercepts_[None, :]


class AdaptiveLassoPathRegressor(_AdaptivePathBase):
    """Adaptive lasso (Zou 2006) along a λ-path with warm starts.

    Pilot is a plain lasso fit (MCP at γ = 1e9); final is also lasso
    with the per-feature inverse-magnitude weights. With pilot magnitudes
    η-rescaled, the final solve produces the asymptotically unbiased
    "oracle" sparse solution under the right regularity conditions.

    Parameters
    ----------
    eta : float, default 1.0
        Adaptive-weight exponent. `w_j = 1 / max(|β_pilot[j]|, eps)^η`.
    eps_pilot : float, default 1e-6
        Floor on `|β_pilot|` to keep weights finite.
    n_pilot_lambdas : int, default 10
        Length of the pilot's auto λ-grid.
    pilot_position : {'mid', 'last'} or int, default 'mid'
        Which λ along the pilot path to read β from. ``'mid'`` is a
        good default — the path's middle is typically a reasonable
        bias/variance compromise. ``'last'`` is closest to OLS.
    lambdas : array-like or None, default None
        Forwarded to the final estimator (`MCPPathRegressor` at γ = 1e9).
        See its docstring for the rest of the standard kwargs
        (``n_lambdas``, ``lambda_min_ratio``, ``max_iter``, ``tol``,
        ``fit_intercept``, ``standardize``, ``screening``,
        ``acceleration``).

    Attributes
    ----------
    coefs_ : (n_lambdas, n_features)
    intercepts_ : (n_lambdas,)
    lambdas_ : (n_lambdas,)
    coef_pilot_ : (n_features,) — the β read off the pilot path.
    weights_ : (n_features,) — the adaptive weights computed from the pilot.
    info_ : dict
    n_features_in_ : int
    """

    _final_cls = MCPPathRegressor

    def __init__(
        self,
        *,
        eta: float = 1.0,
        eps_pilot: float = 1e-6,
        n_pilot_lambdas: int = 10,
        pilot_position: PilotPosition | int = "mid",
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        screening: str = "strong",
        acceleration: int | None = 5,
    ) -> None:
        self.eta = eta
        self.eps_pilot = eps_pilot
        self.n_pilot_lambdas = n_pilot_lambdas
        self.pilot_position = pilot_position
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.screening = screening
        self.acceleration = acceleration

    def _extra_kwargs(self) -> dict[str, Any]:
        return {"gamma": 1e9}


class AdaptiveMCPPathRegressor(_AdaptivePathBase):
    """Adaptive MCP along a λ-path with warm starts. Pilot is plain
    lasso; final is MCP at the user's `gamma` with pilot-derived weights."""

    _final_cls = MCPPathRegressor

    def __init__(
        self,
        gamma: float = 3.0,
        *,
        eta: float = 1.0,
        eps_pilot: float = 1e-6,
        n_pilot_lambdas: int = 10,
        pilot_position: PilotPosition | int = "mid",
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        screening: str = "strong",
        acceleration: int | None = 5,
    ) -> None:
        self.gamma = gamma
        self.eta = eta
        self.eps_pilot = eps_pilot
        self.n_pilot_lambdas = n_pilot_lambdas
        self.pilot_position = pilot_position
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.screening = screening
        self.acceleration = acceleration

    def _extra_kwargs(self) -> dict[str, Any]:
        return {"gamma": self.gamma}


class AdaptiveSCADPathRegressor(_AdaptivePathBase):
    """Adaptive SCAD along a λ-path with warm starts. Pilot is plain
    lasso; final is SCAD at the user's `a` with pilot-derived weights."""

    _final_cls = SCADPathRegressor

    def __init__(
        self,
        a: float = 3.7,
        *,
        eta: float = 1.0,
        eps_pilot: float = 1e-6,
        n_pilot_lambdas: int = 10,
        pilot_position: PilotPosition | int = "mid",
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        screening: str = "strong",
        acceleration: int | None = 5,
    ) -> None:
        self.a = a
        self.eta = eta
        self.eps_pilot = eps_pilot
        self.n_pilot_lambdas = n_pilot_lambdas
        self.pilot_position = pilot_position
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.screening = screening
        self.acceleration = acceleration

    def _extra_kwargs(self) -> dict[str, Any]:
        return {"a": self.a}


# =========================================================================
# Path-CV estimators (K-fold CV with pilot fit on full data)
# =========================================================================


class _AdaptivePathCVBase(BaseEstimator, RegressorMixin):
    """K-fold CV for the adaptive estimators. Pilot weights are computed
    once on the full data (not per fold) — they're a data-derived
    hyperparameter rather than a model parameter, and the per-fold
    final fit uses these fixed weights. CV picks the λ minimizing
    mean test MSE."""

    coef_: NDArray[np.float64]
    intercept_: float
    coef_pilot_: NDArray[np.float64]
    weights_: NDArray[np.float64]
    cv_scores_: NDArray[np.float64]
    cv_mean_scores_: NDArray[np.float64]
    cv_std_scores_: NDArray[np.float64]
    lambdas_: NDArray[np.float64]
    lambda_best_: float
    info_: dict[str, Any]
    n_features_in_: int

    _final_cls: Any

    def _extra_kwargs(self) -> dict[str, Any]:
        return {}

    def _make_final(self, weights: NDArray[np.float64], **overrides):
        kw: dict[str, Any] = dict(
            lambdas=self.lambdas,  # type: ignore[attr-defined]
            n_lambdas=self.n_lambdas,  # type: ignore[attr-defined]
            lambda_min_ratio=self.lambda_min_ratio,  # type: ignore[attr-defined]
            weights=weights,
            max_iter=self.max_iter,  # type: ignore[attr-defined]
            tol=self.tol,  # type: ignore[attr-defined]
            fit_intercept=self.fit_intercept,  # type: ignore[attr-defined]
            standardize=self.standardize,  # type: ignore[attr-defined]
            screening=self.screening,  # type: ignore[attr-defined]
            acceleration=self.acceleration,  # type: ignore[attr-defined]
        )
        kw.update(self._extra_kwargs())
        kw.update(overrides)
        return self._final_cls(**kw)

    def fit(self, x, y) -> "_AdaptivePathCVBase":
        # Step 1: pilot weights from the full data.
        beta_pilot = _fit_pilot(
            x,
            y,
            self.n_pilot_lambdas,  # type: ignore[attr-defined]
            self.pilot_position,  # type: ignore[attr-defined]
            self.fit_intercept,  # type: ignore[attr-defined]
            self.standardize,  # type: ignore[attr-defined]
        )
        weights = _adaptive_weights(
            beta_pilot,
            self.eta,  # type: ignore[attr-defined]
            self.eps_pilot,  # type: ignore[attr-defined]
        )

        # Step 2: full-data final refit (provides λ-grid and final β).
        full = self._make_final(weights).fit(x, y)
        lambdas = full.lambdas_

        # Step 3: K-fold CV on the final estimator with fixed pilot weights.
        y_arr = np.ascontiguousarray(y, dtype=np.float64)
        if _is_sparse(x):
            from scipy import sparse  # type: ignore[import-untyped]
            x_for_indexing = x.tocsr() if not sparse.isspmatrix_csr(x) else x
        else:
            x_for_indexing = np.ascontiguousarray(x, dtype=np.float64)

        cv = self.cv  # type: ignore[attr-defined]
        random_state = self.random_state  # type: ignore[attr-defined]
        if isinstance(cv, int):
            splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
        else:
            splitter = cv

        n_lambdas = lambdas.shape[0]
        n = y_arr.shape[0]
        split_input = (
            np.zeros((n, 1)) if _is_sparse(x) else x_for_indexing
        )
        scores = np.full((splitter.get_n_splits(split_input), n_lambdas), np.nan)

        for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(split_input)):
            x_tr = x_for_indexing[train_idx]
            x_te = x_for_indexing[test_idx]
            y_tr = y_arr[train_idx]
            y_te = y_arr[test_idx]
            fold = self._make_final(weights, lambdas=lambdas).fit(x_tr, y_tr)
            for lam_idx in range(n_lambdas):
                pred = (
                    x_te @ fold.coefs_[lam_idx] + fold.intercepts_[lam_idx]
                )
                if hasattr(pred, "toarray"):
                    pred = pred.toarray()
                pred = np.asarray(pred).ravel()
                diff = y_te - pred
                scores[fold_idx, lam_idx] = float(np.mean(diff * diff))

        self.cv_scores_ = scores
        self.cv_mean_scores_ = np.nanmean(scores, axis=0)
        self.cv_std_scores_ = np.nanstd(scores, axis=0)
        best = int(np.argmin(self.cv_mean_scores_))
        self.lambdas_ = lambdas
        self.lambda_best_ = float(lambdas[best])
        self.coef_ = full.coefs_[best]
        self.intercept_ = float(full.intercepts_[best])
        self.coef_pilot_ = beta_pilot
        self.weights_ = weights
        self.info_ = full.info_
        self.n_features_in_ = full.n_features_in_
        return self

    def predict(self, x) -> NDArray[np.float64]:
        if _is_sparse(x):
            pred = x @ self.coef_
            if hasattr(pred, "toarray"):
                pred = pred.toarray()
            return np.asarray(pred) + self.intercept_
        x = np.ascontiguousarray(x, dtype=np.float64)
        return x @ self.coef_ + self.intercept_


class AdaptiveLassoPathCV(_AdaptivePathCVBase):
    """K-fold CV over an adaptive-lasso λ-path."""

    _final_cls = MCPPathRegressor

    def __init__(
        self,
        *,
        eta: float = 1.0,
        eps_pilot: float = 1e-6,
        n_pilot_lambdas: int = 10,
        pilot_position: PilotPosition | int = "mid",
        cv: Any = 5,
        random_state: int | None = None,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        screening: str = "strong",
        acceleration: int | None = 5,
    ) -> None:
        self.eta = eta
        self.eps_pilot = eps_pilot
        self.n_pilot_lambdas = n_pilot_lambdas
        self.pilot_position = pilot_position
        self.cv = cv
        self.random_state = random_state
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.screening = screening
        self.acceleration = acceleration

    def _extra_kwargs(self) -> dict[str, Any]:
        return {"gamma": 1e9}


class AdaptiveMCPPathCV(_AdaptivePathCVBase):
    """K-fold CV over an adaptive-MCP λ-path."""

    _final_cls = MCPPathRegressor

    def __init__(
        self,
        gamma: float = 3.0,
        *,
        eta: float = 1.0,
        eps_pilot: float = 1e-6,
        n_pilot_lambdas: int = 10,
        pilot_position: PilotPosition | int = "mid",
        cv: Any = 5,
        random_state: int | None = None,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        screening: str = "strong",
        acceleration: int | None = 5,
    ) -> None:
        self.gamma = gamma
        self.eta = eta
        self.eps_pilot = eps_pilot
        self.n_pilot_lambdas = n_pilot_lambdas
        self.pilot_position = pilot_position
        self.cv = cv
        self.random_state = random_state
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.screening = screening
        self.acceleration = acceleration

    def _extra_kwargs(self) -> dict[str, Any]:
        return {"gamma": self.gamma}


class AdaptiveSCADPathCV(_AdaptivePathCVBase):
    """K-fold CV over an adaptive-SCAD λ-path."""

    _final_cls = SCADPathRegressor

    def __init__(
        self,
        a: float = 3.7,
        *,
        eta: float = 1.0,
        eps_pilot: float = 1e-6,
        n_pilot_lambdas: int = 10,
        pilot_position: PilotPosition | int = "mid",
        cv: Any = 5,
        random_state: int | None = None,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        screening: str = "strong",
        acceleration: int | None = 5,
    ) -> None:
        self.a = a
        self.eta = eta
        self.eps_pilot = eps_pilot
        self.n_pilot_lambdas = n_pilot_lambdas
        self.pilot_position = pilot_position
        self.cv = cv
        self.random_state = random_state
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.screening = screening
        self.acceleration = acceleration

    def _extra_kwargs(self) -> dict[str, Any]:
        return {"a": self.a}


# =========================================================================
# Adaptive group estimators (LS only; pilot is GroupLassoPathRegressor)
# =========================================================================


def _fit_group_pilot(
    x,
    y,
    groups: NDArray[np.int64],
    n_pilot_lambdas: int,
    pilot_position,
    fit_intercept: bool,
    standardize: bool,
) -> NDArray[np.float64]:
    """Run a plain group lasso path on `(X, y)` and return β at the
    resolved `pilot_position`."""
    pilot = GroupLassoPathRegressor(
        groups=groups,
        n_lambdas=n_pilot_lambdas,
        lambda_min_ratio=1e-3,
        fit_intercept=fit_intercept,
        standardize=standardize,
    ).fit(x, y)
    idx = _validate_pilot_position(pilot_position, len(pilot.lambdas_))
    return pilot.coefs_[idx].copy()


class _AdaptiveGroupPathBase(BaseEstimator, RegressorMixin):
    """Common scaffold for adaptive group-penalty path estimators.
    Subclasses set `_final_cls` and `_extra_kwargs()` for penalty params."""

    coefs_: NDArray[np.float64]
    intercepts_: NDArray[np.float64]
    lambdas_: NDArray[np.float64]
    coef_pilot_: NDArray[np.float64]
    weights_: NDArray[np.float64]
    info_: dict[str, Any]
    n_features_in_: int

    _final_cls: Any

    def _extra_kwargs(self) -> dict[str, Any]:
        return {}

    def _make_final(self, weights: NDArray[np.float64]):
        kw: dict[str, Any] = dict(
            groups=self.groups,  # type: ignore[attr-defined]
            lambdas=self.lambdas,  # type: ignore[attr-defined]
            n_lambdas=self.n_lambdas,  # type: ignore[attr-defined]
            lambda_min_ratio=self.lambda_min_ratio,  # type: ignore[attr-defined]
            weights=weights,
            max_iter=self.max_iter,  # type: ignore[attr-defined]
            tol=self.tol,  # type: ignore[attr-defined]
            fit_intercept=self.fit_intercept,  # type: ignore[attr-defined]
            standardize=self.standardize,  # type: ignore[attr-defined]
            screening=self.screening,  # type: ignore[attr-defined]
            acceleration=self.acceleration,  # type: ignore[attr-defined]
            parallel=self.parallel,  # type: ignore[attr-defined]
        )
        kw.update(self._extra_kwargs())
        return self._final_cls(**kw)

    def fit(self, x, y) -> "_AdaptiveGroupPathBase":
        beta_pilot = _fit_group_pilot(
            x,
            y,
            self.groups,  # type: ignore[attr-defined]
            self.n_pilot_lambdas,  # type: ignore[attr-defined]
            self.pilot_position,  # type: ignore[attr-defined]
            self.fit_intercept,  # type: ignore[attr-defined]
            self.standardize,  # type: ignore[attr-defined]
        )
        weights = _adaptive_weights_per_group(
            beta_pilot,
            self.groups,  # type: ignore[attr-defined]
            self.eta,  # type: ignore[attr-defined]
            self.eps_pilot,  # type: ignore[attr-defined]
        )
        final = self._make_final(weights).fit(x, y)
        self.coefs_ = final.coefs_
        self.intercepts_ = final.intercepts_
        self.lambdas_ = final.lambdas_
        self.coef_pilot_ = beta_pilot
        self.weights_ = weights
        self.info_ = final.info_
        self.n_features_in_ = final.n_features_in_
        return self

    def predict(self, x) -> NDArray[np.float64]:
        if _is_sparse(x):
            return np.asarray(x @ self.coefs_.T) + self.intercepts_[None, :]
        x = np.ascontiguousarray(x, dtype=np.float64)
        return x @ self.coefs_.T + self.intercepts_[None, :]


class AdaptiveGroupLassoPathRegressor(_AdaptiveGroupPathBase):
    """Adaptive group lasso along a λ-path. Pilot is plain group lasso;
    final is also group lasso with per-group inverse-norm weights
    `w_g = 1 / max(‖β_pilot[g]‖_2, ε)^η`."""

    _final_cls = GroupLassoPathRegressor

    def __init__(
        self,
        groups: NDArray[np.int64],
        *,
        eta: float = 1.0,
        eps_pilot: float = 1e-6,
        n_pilot_lambdas: int = 10,
        pilot_position: PilotPosition | int = "mid",
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        screening: str = "strong",
        acceleration: int | None = 5,
        parallel: bool = False,
    ) -> None:
        self.groups = groups
        self.eta = eta
        self.eps_pilot = eps_pilot
        self.n_pilot_lambdas = n_pilot_lambdas
        self.pilot_position = pilot_position
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.screening = screening
        self.acceleration = acceleration
        self.parallel = parallel


class AdaptiveGroupMCPPathRegressor(_AdaptiveGroupPathBase):
    """Adaptive group MCP along a λ-path. Pilot is plain group lasso;
    final is group MCP at the user's `gamma` with per-group adaptive
    weights."""

    _final_cls = GroupMCPPathRegressor

    def __init__(
        self,
        groups: NDArray[np.int64],
        gamma: float = 3.0,
        *,
        eta: float = 1.0,
        eps_pilot: float = 1e-6,
        n_pilot_lambdas: int = 10,
        pilot_position: PilotPosition | int = "mid",
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
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
        self.groups = groups
        self.gamma = gamma
        self.eta = eta
        self.eps_pilot = eps_pilot
        self.n_pilot_lambdas = n_pilot_lambdas
        self.pilot_position = pilot_position
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.max_outer = max_outer
        self.outer_tol = outer_tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.screening = screening
        self.acceleration = acceleration
        self.parallel = parallel

    def _extra_kwargs(self) -> dict[str, Any]:
        return {
            "gamma": self.gamma,
            "max_outer": self.max_outer,
            "outer_tol": self.outer_tol,
        }


class AdaptiveGroupSCADPathRegressor(_AdaptiveGroupPathBase):
    """Adaptive group SCAD along a λ-path. Pilot is plain group lasso;
    final is group SCAD at the user's `a` with per-group adaptive
    weights `w_g = 1 / max(‖β_pilot[g]‖_2, ε)^η`."""

    _final_cls = GroupSCADPathRegressor

    def __init__(
        self,
        groups: NDArray[np.int64],
        a: float = 3.7,
        *,
        eta: float = 1.0,
        eps_pilot: float = 1e-6,
        n_pilot_lambdas: int = 10,
        pilot_position: PilotPosition | int = "mid",
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
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
        self.groups = groups
        self.a = a
        self.eta = eta
        self.eps_pilot = eps_pilot
        self.n_pilot_lambdas = n_pilot_lambdas
        self.pilot_position = pilot_position
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.max_outer = max_outer
        self.outer_tol = outer_tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.screening = screening
        self.acceleration = acceleration
        self.parallel = parallel

    def _extra_kwargs(self) -> dict[str, Any]:
        return {
            "a": self.a,
            "max_outer": self.max_outer,
            "outer_tol": self.outer_tol,
        }


class _AdaptiveGroupPathCVBase(BaseEstimator, RegressorMixin):
    """K-fold CV for adaptive group estimators. Pilot weights computed
    once on full data; per-fold final fit uses fixed weights."""

    coef_: NDArray[np.float64]
    intercept_: float
    coef_pilot_: NDArray[np.float64]
    weights_: NDArray[np.float64]
    cv_scores_: NDArray[np.float64]
    cv_mean_scores_: NDArray[np.float64]
    cv_std_scores_: NDArray[np.float64]
    lambdas_: NDArray[np.float64]
    lambda_best_: float
    info_: dict[str, Any]
    n_features_in_: int

    _final_cls: Any

    def _extra_kwargs(self) -> dict[str, Any]:
        return {}

    def _make_final(self, weights: NDArray[np.float64], **overrides):
        kw: dict[str, Any] = dict(
            groups=self.groups,  # type: ignore[attr-defined]
            lambdas=self.lambdas,  # type: ignore[attr-defined]
            n_lambdas=self.n_lambdas,  # type: ignore[attr-defined]
            lambda_min_ratio=self.lambda_min_ratio,  # type: ignore[attr-defined]
            weights=weights,
            max_iter=self.max_iter,  # type: ignore[attr-defined]
            tol=self.tol,  # type: ignore[attr-defined]
            fit_intercept=self.fit_intercept,  # type: ignore[attr-defined]
            standardize=self.standardize,  # type: ignore[attr-defined]
            screening=self.screening,  # type: ignore[attr-defined]
            acceleration=self.acceleration,  # type: ignore[attr-defined]
            parallel=self.parallel,  # type: ignore[attr-defined]
        )
        kw.update(self._extra_kwargs())
        kw.update(overrides)
        return self._final_cls(**kw)

    def fit(self, x, y) -> "_AdaptiveGroupPathCVBase":
        beta_pilot = _fit_group_pilot(
            x, y,
            self.groups,  # type: ignore[attr-defined]
            self.n_pilot_lambdas,  # type: ignore[attr-defined]
            self.pilot_position,  # type: ignore[attr-defined]
            self.fit_intercept,  # type: ignore[attr-defined]
            self.standardize,  # type: ignore[attr-defined]
        )
        weights = _adaptive_weights_per_group(
            beta_pilot,
            self.groups,  # type: ignore[attr-defined]
            self.eta,  # type: ignore[attr-defined]
            self.eps_pilot,  # type: ignore[attr-defined]
        )

        full = self._make_final(weights).fit(x, y)
        lambdas = full.lambdas_

        y_arr = np.ascontiguousarray(y, dtype=np.float64)
        if _is_sparse(x):
            from scipy import sparse  # type: ignore[import-untyped]
            x_for_indexing = x.tocsr() if not sparse.isspmatrix_csr(x) else x
        else:
            x_for_indexing = np.ascontiguousarray(x, dtype=np.float64)

        cv = self.cv  # type: ignore[attr-defined]
        random_state = self.random_state  # type: ignore[attr-defined]
        if isinstance(cv, int):
            splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
        else:
            splitter = cv

        n_lambdas = lambdas.shape[0]
        n = y_arr.shape[0]
        split_input = np.zeros((n, 1)) if _is_sparse(x) else x_for_indexing
        scores = np.full((splitter.get_n_splits(split_input), n_lambdas), np.nan)

        for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(split_input)):
            x_tr = x_for_indexing[train_idx]
            x_te = x_for_indexing[test_idx]
            y_tr = y_arr[train_idx]
            y_te = y_arr[test_idx]
            fold = self._make_final(weights, lambdas=lambdas).fit(x_tr, y_tr)
            for lam_idx in range(n_lambdas):
                pred = x_te @ fold.coefs_[lam_idx] + fold.intercepts_[lam_idx]
                if hasattr(pred, "toarray"):
                    pred = pred.toarray()
                pred = np.asarray(pred).ravel()
                diff = y_te - pred
                scores[fold_idx, lam_idx] = float(np.mean(diff * diff))

        self.cv_scores_ = scores
        self.cv_mean_scores_ = np.nanmean(scores, axis=0)
        self.cv_std_scores_ = np.nanstd(scores, axis=0)
        best = int(np.argmin(self.cv_mean_scores_))
        self.lambdas_ = lambdas
        self.lambda_best_ = float(lambdas[best])
        self.coef_ = full.coefs_[best]
        self.intercept_ = float(full.intercepts_[best])
        self.coef_pilot_ = beta_pilot
        self.weights_ = weights
        self.info_ = full.info_
        self.n_features_in_ = full.n_features_in_
        return self

    def predict(self, x) -> NDArray[np.float64]:
        if _is_sparse(x):
            pred = x @ self.coef_
            if hasattr(pred, "toarray"):
                pred = pred.toarray()
            return np.asarray(pred) + self.intercept_
        x = np.ascontiguousarray(x, dtype=np.float64)
        return x @ self.coef_ + self.intercept_


class AdaptiveGroupLassoPathCV(_AdaptiveGroupPathCVBase):
    """K-fold CV over an adaptive-group-lasso λ-path."""

    _final_cls = GroupLassoPathRegressor

    def __init__(
        self,
        groups: NDArray[np.int64],
        *,
        eta: float = 1.0,
        eps_pilot: float = 1e-6,
        n_pilot_lambdas: int = 10,
        pilot_position: PilotPosition | int = "mid",
        cv: Any = 5,
        random_state: int | None = None,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        screening: str = "strong",
        acceleration: int | None = 5,
        parallel: bool = False,
    ) -> None:
        self.groups = groups
        self.eta = eta
        self.eps_pilot = eps_pilot
        self.n_pilot_lambdas = n_pilot_lambdas
        self.pilot_position = pilot_position
        self.cv = cv
        self.random_state = random_state
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.screening = screening
        self.acceleration = acceleration
        self.parallel = parallel


class AdaptiveGroupMCPPathCV(_AdaptiveGroupPathCVBase):
    """K-fold CV over an adaptive-group-MCP λ-path."""

    _final_cls = GroupMCPPathRegressor

    def __init__(
        self,
        groups: NDArray[np.int64],
        gamma: float = 3.0,
        *,
        eta: float = 1.0,
        eps_pilot: float = 1e-6,
        n_pilot_lambdas: int = 10,
        pilot_position: PilotPosition | int = "mid",
        cv: Any = 5,
        random_state: int | None = None,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
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
        self.groups = groups
        self.gamma = gamma
        self.eta = eta
        self.eps_pilot = eps_pilot
        self.n_pilot_lambdas = n_pilot_lambdas
        self.pilot_position = pilot_position
        self.cv = cv
        self.random_state = random_state
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.max_outer = max_outer
        self.outer_tol = outer_tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.screening = screening
        self.acceleration = acceleration
        self.parallel = parallel

    def _extra_kwargs(self) -> dict[str, Any]:
        return {
            "gamma": self.gamma,
            "max_outer": self.max_outer,
            "outer_tol": self.outer_tol,
        }


class AdaptiveGroupSCADPathCV(_AdaptiveGroupPathCVBase):
    """K-fold CV over an adaptive-group-SCAD λ-path."""

    _final_cls = GroupSCADPathRegressor

    def __init__(
        self,
        groups: NDArray[np.int64],
        a: float = 3.7,
        *,
        eta: float = 1.0,
        eps_pilot: float = 1e-6,
        n_pilot_lambdas: int = 10,
        pilot_position: PilotPosition | int = "mid",
        cv: Any = 5,
        random_state: int | None = None,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
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
        self.groups = groups
        self.a = a
        self.eta = eta
        self.eps_pilot = eps_pilot
        self.n_pilot_lambdas = n_pilot_lambdas
        self.pilot_position = pilot_position
        self.cv = cv
        self.random_state = random_state
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.max_outer = max_outer
        self.outer_tol = outer_tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.screening = screening
        self.acceleration = acceleration
        self.parallel = parallel

    def _extra_kwargs(self) -> dict[str, Any]:
        return {
            "a": self.a,
            "max_outer": self.max_outer,
            "outer_tol": self.outer_tol,
        }


# =========================================================================
# Adaptive GLM estimators (Logistic, Poisson, Cox × Lasso, MCP, SCAD)
# =========================================================================
#
# Pilot is the corresponding GLM's lasso path (e.g.,
# `LogisticLassoPathRegressor` for binomial). Final is the user's
# chosen GLM-penalty path with adaptive weights derived from the pilot.
#
# Path classes store the fitted final estimator as `_final_estimator_`
# and delegate `predict_proba` / `decision_function` / `predict` to it
# (so the GLM-specific predict semantics — binary class label for
# logistic, μ = exp(η) for Poisson, η for Cox — are inherited).
#
# CV classes extend the existing per-family CV mixins
# (`_LogisticPathCVMixin`, `_PoissonPathCVMixin`, `_CoxPathCVMixin`),
# overriding `fit` to compute pilot weights once on the full data and
# `_make_base_path` to inject those weights into every per-fold
# refit. Cox uses `fit(x, time, event)` and StratifiedKFold by event.


def _fit_pilot_logistic(
    x,
    y,
    n_pilot_lambdas: int,
    pilot_position,
    fit_intercept: bool,
    standardize: bool,
) -> NDArray[np.float64]:
    pilot = LogisticLassoPathRegressor(
        n_lambdas=n_pilot_lambdas,
        lambda_min_ratio=1e-3,
        fit_intercept=fit_intercept,
        standardize=standardize,
    ).fit(x, y)
    idx = _validate_pilot_position(pilot_position, len(pilot.lambdas_))
    return pilot.coefs_[idx].copy()


def _fit_pilot_poisson(
    x,
    y,
    n_pilot_lambdas: int,
    pilot_position,
    fit_intercept: bool,
    standardize: bool,
) -> NDArray[np.float64]:
    pilot = PoissonLassoPathRegressor(
        n_lambdas=n_pilot_lambdas,
        lambda_min_ratio=1e-3,
        fit_intercept=fit_intercept,
        standardize=standardize,
    ).fit(x, y)
    idx = _validate_pilot_position(pilot_position, len(pilot.lambdas_))
    return pilot.coefs_[idx].copy()


def _fit_pilot_cox(
    x,
    time,
    event,
    n_pilot_lambdas: int,
    pilot_position,
    standardize: bool,
) -> NDArray[np.float64]:
    pilot = CoxMCPPathRegressor(
        gamma=1e9,
        n_lambdas=n_pilot_lambdas,
        lambda_min_ratio=1e-3,
        standardize=standardize,
    ).fit(x, time, event)
    idx = _validate_pilot_position(pilot_position, len(pilot.lambdas_))
    return pilot.coefs_[idx].copy()


# ---- Logistic ----------------------------------------------------------


class _AdaptiveLogisticPathBase(BaseEstimator, ClassifierMixin):
    """Adaptive logistic path: pilot lasso → adaptive weights → final fit.
    Subclass sets `_final_cls` and `_extra_kwargs()`."""

    coefs_: NDArray[np.float64]
    intercepts_: NDArray[np.float64]
    lambdas_: NDArray[np.float64]
    coef_pilot_: NDArray[np.float64]
    weights_: NDArray[np.float64]
    info_: dict[str, Any]
    n_features_in_: int

    _final_cls: Any

    def _extra_kwargs(self) -> dict[str, Any]:
        return {}

    def _make_final(self, weights: NDArray[np.float64]):
        kw: dict[str, Any] = dict(
            lambdas=self.lambdas,  # type: ignore[attr-defined]
            n_lambdas=self.n_lambdas,  # type: ignore[attr-defined]
            lambda_min_ratio=self.lambda_min_ratio,  # type: ignore[attr-defined]
            weights=weights,
            max_iter=self.max_iter,  # type: ignore[attr-defined]
            tol=self.tol,  # type: ignore[attr-defined]
            fit_intercept=self.fit_intercept,  # type: ignore[attr-defined]
            standardize=self.standardize,  # type: ignore[attr-defined]
            acceleration=self.acceleration,  # type: ignore[attr-defined]
            max_outer=self.max_outer,  # type: ignore[attr-defined]
            outer_tol=self.outer_tol,  # type: ignore[attr-defined]
        )
        kw.update(self._extra_kwargs())
        return self._final_cls(**kw)

    def fit(self, x, y) -> "_AdaptiveLogisticPathBase":
        beta_pilot = _fit_pilot_logistic(
            x, y,
            self.n_pilot_lambdas,  # type: ignore[attr-defined]
            self.pilot_position,  # type: ignore[attr-defined]
            self.fit_intercept,  # type: ignore[attr-defined]
            self.standardize,  # type: ignore[attr-defined]
        )
        weights = _adaptive_weights(
            beta_pilot,
            self.eta,  # type: ignore[attr-defined]
            self.eps_pilot,  # type: ignore[attr-defined]
        )
        final = self._make_final(weights).fit(x, y)
        self._final_estimator_ = final
        self.coefs_ = final.coefs_
        self.intercepts_ = final.intercepts_
        self.lambdas_ = final.lambdas_
        self.coef_pilot_ = beta_pilot
        self.weights_ = weights
        self.info_ = final.info_
        self.n_features_in_ = final.n_features_in_
        return self

    def decision_function(self, x):
        return self._final_estimator_.decision_function(x)

    def predict_proba(self, x):
        return self._final_estimator_.predict_proba(x)

    def predict(self, x):
        return self._final_estimator_.predict(x)


def _logistic_path_init(self, gamma_or_a, *, eta, eps_pilot, n_pilot_lambdas,
                         pilot_position, lambdas, n_lambdas, lambda_min_ratio,
                         max_iter, tol, max_outer, outer_tol, fit_intercept,
                         standardize, acceleration, gamma_or_a_attr):
    setattr(self, gamma_or_a_attr, gamma_or_a)
    self.eta = eta
    self.eps_pilot = eps_pilot
    self.n_pilot_lambdas = n_pilot_lambdas
    self.pilot_position = pilot_position
    self.lambdas = lambdas
    self.n_lambdas = n_lambdas
    self.lambda_min_ratio = lambda_min_ratio
    self.max_iter = max_iter
    self.tol = tol
    self.max_outer = max_outer
    self.outer_tol = outer_tol
    self.fit_intercept = fit_intercept
    self.standardize = standardize
    self.acceleration = acceleration


class AdaptiveLogisticLassoPathRegressor(_AdaptiveLogisticPathBase):
    """Adaptive logistic lasso (pilot lasso + adaptive lasso final)."""

    _final_cls = LogisticLassoPathRegressor

    def __init__(
        self,
        *,
        eta: float = 1.0,
        eps_pilot: float = 1e-6,
        n_pilot_lambdas: int = 10,
        pilot_position: PilotPosition | int = "mid",
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        max_iter: int = 100,
        tol: float = 1e-6,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
    ) -> None:
        self.eta = eta
        self.eps_pilot = eps_pilot
        self.n_pilot_lambdas = n_pilot_lambdas
        self.pilot_position = pilot_position
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.max_outer = max_outer
        self.outer_tol = outer_tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.acceleration = acceleration


class AdaptiveLogisticMCPPathRegressor(_AdaptiveLogisticPathBase):
    """Adaptive logistic MCP."""

    _final_cls = LogisticMCPPathRegressor

    def __init__(
        self,
        gamma: float = 3.0,
        *,
        eta: float = 1.0,
        eps_pilot: float = 1e-6,
        n_pilot_lambdas: int = 10,
        pilot_position: PilotPosition | int = "mid",
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        max_iter: int = 100,
        tol: float = 1e-6,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
    ) -> None:
        self.gamma = gamma
        self.eta = eta
        self.eps_pilot = eps_pilot
        self.n_pilot_lambdas = n_pilot_lambdas
        self.pilot_position = pilot_position
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.max_outer = max_outer
        self.outer_tol = outer_tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.acceleration = acceleration

    def _extra_kwargs(self) -> dict[str, Any]:
        return {"gamma": self.gamma}


class AdaptiveLogisticSCADPathRegressor(_AdaptiveLogisticPathBase):
    """Adaptive logistic SCAD."""

    _final_cls = LogisticSCADPathRegressor

    def __init__(
        self,
        a: float = 3.7,
        *,
        eta: float = 1.0,
        eps_pilot: float = 1e-6,
        n_pilot_lambdas: int = 10,
        pilot_position: PilotPosition | int = "mid",
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        max_iter: int = 100,
        tol: float = 1e-6,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
    ) -> None:
        self.a = a
        self.eta = eta
        self.eps_pilot = eps_pilot
        self.n_pilot_lambdas = n_pilot_lambdas
        self.pilot_position = pilot_position
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.max_outer = max_outer
        self.outer_tol = outer_tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.acceleration = acceleration

    def _extra_kwargs(self) -> dict[str, Any]:
        return {"a": self.a}


class _AdaptiveLogisticPathCVBase(_LogisticPathCVMixin, BaseEstimator, ClassifierMixin):
    """Logistic adaptive CV: pilot weights once on full data, K-fold CV
    on the final fit with those fixed weights. Subclass sets `_final_cls`
    and `_extra_kwargs()`."""

    coef_pilot_: NDArray[np.float64]
    weights_: NDArray[np.float64]
    _final_cls: Any

    def _extra_kwargs(self) -> dict[str, Any]:
        return {}

    def _make_base_path(self, **overrides):
        kw: dict[str, Any] = dict(
            lambdas=self.lambdas,
            n_lambdas=self.n_lambdas,
            lambda_min_ratio=self.lambda_min_ratio,
            weights=self.weights_,
            max_iter=self.max_iter,
            tol=self.tol,
            max_outer=self.max_outer,
            outer_tol=self.outer_tol,
            fit_intercept=self.fit_intercept,
            standardize=self.standardize,
            acceleration=self.acceleration,
        )
        kw.update(self._extra_kwargs())
        kw.update(overrides)
        return self._final_cls(**kw)

    def fit(self, x, y):
        beta_pilot = _fit_pilot_logistic(
            x, y,
            self.n_pilot_lambdas,
            self.pilot_position,
            self.fit_intercept,
            self.standardize,
        )
        self.coef_pilot_ = beta_pilot
        self.weights_ = _adaptive_weights(beta_pilot, self.eta, self.eps_pilot)
        return super().fit(x, y)


def _adaptive_logistic_cv_init(
    self, gamma_or_a, gamma_or_a_attr, *, eta, eps_pilot, n_pilot_lambdas,
    pilot_position, cv, random_state, lambdas, n_lambdas, lambda_min_ratio,
    max_iter, tol, max_outer, outer_tol, fit_intercept, standardize, acceleration,
):
    setattr(self, gamma_or_a_attr, gamma_or_a)
    self.eta = eta
    self.eps_pilot = eps_pilot
    self.n_pilot_lambdas = n_pilot_lambdas
    self.pilot_position = pilot_position
    self.cv = cv
    self.random_state = random_state
    self.lambdas = lambdas
    self.n_lambdas = n_lambdas
    self.lambda_min_ratio = lambda_min_ratio
    self.max_iter = max_iter
    self.tol = tol
    self.max_outer = max_outer
    self.outer_tol = outer_tol
    self.fit_intercept = fit_intercept
    self.standardize = standardize
    self.acceleration = acceleration


class AdaptiveLogisticLassoPathCV(_AdaptiveLogisticPathCVBase):
    """K-fold CV over an adaptive logistic-lasso path."""

    _final_cls = LogisticLassoPathRegressor

    def __init__(
        self,
        *,
        eta: float = 1.0,
        eps_pilot: float = 1e-6,
        n_pilot_lambdas: int = 10,
        pilot_position: PilotPosition | int = "mid",
        cv: Any = 5,
        random_state: int | None = None,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        max_iter: int = 100,
        tol: float = 1e-6,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
    ) -> None:
        self.eta = eta
        self.eps_pilot = eps_pilot
        self.n_pilot_lambdas = n_pilot_lambdas
        self.pilot_position = pilot_position
        self.cv = cv
        self.random_state = random_state
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.max_outer = max_outer
        self.outer_tol = outer_tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.acceleration = acceleration


class AdaptiveLogisticMCPPathCV(_AdaptiveLogisticPathCVBase):
    """K-fold CV over an adaptive logistic-MCP path."""

    _final_cls = LogisticMCPPathRegressor

    def __init__(
        self,
        gamma: float = 3.0,
        *,
        eta: float = 1.0,
        eps_pilot: float = 1e-6,
        n_pilot_lambdas: int = 10,
        pilot_position: PilotPosition | int = "mid",
        cv: Any = 5,
        random_state: int | None = None,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        max_iter: int = 100,
        tol: float = 1e-6,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
    ) -> None:
        _adaptive_logistic_cv_init(
            self, gamma, "gamma", eta=eta, eps_pilot=eps_pilot,
            n_pilot_lambdas=n_pilot_lambdas, pilot_position=pilot_position,
            cv=cv, random_state=random_state, lambdas=lambdas,
            n_lambdas=n_lambdas, lambda_min_ratio=lambda_min_ratio,
            max_iter=max_iter, tol=tol, max_outer=max_outer, outer_tol=outer_tol,
            fit_intercept=fit_intercept, standardize=standardize,
            acceleration=acceleration,
        )

    def _extra_kwargs(self) -> dict[str, Any]:
        return {"gamma": self.gamma}


class AdaptiveLogisticSCADPathCV(_AdaptiveLogisticPathCVBase):
    """K-fold CV over an adaptive logistic-SCAD path."""

    _final_cls = LogisticSCADPathRegressor

    def __init__(
        self,
        a: float = 3.7,
        *,
        eta: float = 1.0,
        eps_pilot: float = 1e-6,
        n_pilot_lambdas: int = 10,
        pilot_position: PilotPosition | int = "mid",
        cv: Any = 5,
        random_state: int | None = None,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        max_iter: int = 100,
        tol: float = 1e-6,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
    ) -> None:
        _adaptive_logistic_cv_init(
            self, a, "a", eta=eta, eps_pilot=eps_pilot,
            n_pilot_lambdas=n_pilot_lambdas, pilot_position=pilot_position,
            cv=cv, random_state=random_state, lambdas=lambdas,
            n_lambdas=n_lambdas, lambda_min_ratio=lambda_min_ratio,
            max_iter=max_iter, tol=tol, max_outer=max_outer, outer_tol=outer_tol,
            fit_intercept=fit_intercept, standardize=standardize,
            acceleration=acceleration,
        )

    def _extra_kwargs(self) -> dict[str, Any]:
        return {"a": self.a}


# ---- Poisson -----------------------------------------------------------


class _AdaptivePoissonPathBase(BaseEstimator, RegressorMixin):
    """Adaptive Poisson path: same pattern as logistic but uses
    `PoissonLassoPathRegressor` / `PoissonMCPPathRegressor` /
    `PoissonSCADPathRegressor` for the final."""

    coefs_: NDArray[np.float64]
    intercepts_: NDArray[np.float64]
    lambdas_: NDArray[np.float64]
    coef_pilot_: NDArray[np.float64]
    weights_: NDArray[np.float64]
    info_: dict[str, Any]
    n_features_in_: int

    _final_cls: Any

    def _extra_kwargs(self) -> dict[str, Any]:
        return {}

    def _make_final(self, weights):
        kw: dict[str, Any] = dict(
            lambdas=self.lambdas, n_lambdas=self.n_lambdas,
            lambda_min_ratio=self.lambda_min_ratio, weights=weights,
            max_iter=self.max_iter, tol=self.tol,
            fit_intercept=self.fit_intercept, standardize=self.standardize,
            acceleration=self.acceleration,
            max_outer=self.max_outer, outer_tol=self.outer_tol,
        )
        kw.update(self._extra_kwargs())
        return self._final_cls(**kw)

    def fit(self, x, y) -> "_AdaptivePoissonPathBase":
        beta_pilot = _fit_pilot_poisson(
            x, y, self.n_pilot_lambdas, self.pilot_position,
            self.fit_intercept, self.standardize,
        )
        weights = _adaptive_weights(beta_pilot, self.eta, self.eps_pilot)
        final = self._make_final(weights).fit(x, y)
        self._final_estimator_ = final
        self.coefs_ = final.coefs_
        self.intercepts_ = final.intercepts_
        self.lambdas_ = final.lambdas_
        self.coef_pilot_ = beta_pilot
        self.weights_ = weights
        self.info_ = final.info_
        self.n_features_in_ = final.n_features_in_
        return self

    def decision_function(self, x):
        return self._final_estimator_.decision_function(x)

    def predict(self, x):
        return self._final_estimator_.predict(x)


class AdaptivePoissonLassoPathRegressor(_AdaptivePoissonPathBase):
    """Adaptive Poisson lasso."""

    _final_cls = PoissonLassoPathRegressor

    def __init__(
        self,
        *,
        eta: float = 1.0,
        eps_pilot: float = 1e-6,
        n_pilot_lambdas: int = 10,
        pilot_position: PilotPosition | int = "mid",
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        max_iter: int = 100,
        tol: float = 1e-6,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
    ) -> None:
        self.eta = eta
        self.eps_pilot = eps_pilot
        self.n_pilot_lambdas = n_pilot_lambdas
        self.pilot_position = pilot_position
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.max_outer = max_outer
        self.outer_tol = outer_tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.acceleration = acceleration


class AdaptivePoissonMCPPathRegressor(_AdaptivePoissonPathBase):
    """Adaptive Poisson MCP."""

    _final_cls = PoissonMCPPathRegressor

    def __init__(
        self,
        gamma: float = 3.0,
        *,
        eta: float = 1.0,
        eps_pilot: float = 1e-6,
        n_pilot_lambdas: int = 10,
        pilot_position: PilotPosition | int = "mid",
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        max_iter: int = 100,
        tol: float = 1e-6,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
    ) -> None:
        self.gamma = gamma
        self.eta = eta
        self.eps_pilot = eps_pilot
        self.n_pilot_lambdas = n_pilot_lambdas
        self.pilot_position = pilot_position
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.max_outer = max_outer
        self.outer_tol = outer_tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.acceleration = acceleration

    def _extra_kwargs(self) -> dict[str, Any]:
        return {"gamma": self.gamma}


class AdaptivePoissonSCADPathRegressor(_AdaptivePoissonPathBase):
    """Adaptive Poisson SCAD."""

    _final_cls = PoissonSCADPathRegressor

    def __init__(
        self,
        a: float = 3.7,
        *,
        eta: float = 1.0,
        eps_pilot: float = 1e-6,
        n_pilot_lambdas: int = 10,
        pilot_position: PilotPosition | int = "mid",
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        max_iter: int = 100,
        tol: float = 1e-6,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
    ) -> None:
        self.a = a
        self.eta = eta
        self.eps_pilot = eps_pilot
        self.n_pilot_lambdas = n_pilot_lambdas
        self.pilot_position = pilot_position
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.max_outer = max_outer
        self.outer_tol = outer_tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.acceleration = acceleration

    def _extra_kwargs(self) -> dict[str, Any]:
        return {"a": self.a}


class _AdaptivePoissonPathCVBase(_PoissonPathCVMixin, BaseEstimator, RegressorMixin):
    coef_pilot_: NDArray[np.float64]
    weights_: NDArray[np.float64]
    _final_cls: Any

    def _extra_kwargs(self) -> dict[str, Any]:
        return {}

    def _make_base_path(self, **overrides):
        kw: dict[str, Any] = dict(
            lambdas=self.lambdas, n_lambdas=self.n_lambdas,
            lambda_min_ratio=self.lambda_min_ratio, weights=self.weights_,
            max_iter=self.max_iter, tol=self.tol, max_outer=self.max_outer,
            outer_tol=self.outer_tol, fit_intercept=self.fit_intercept,
            standardize=self.standardize, acceleration=self.acceleration,
        )
        kw.update(self._extra_kwargs())
        kw.update(overrides)
        return self._final_cls(**kw)

    def fit(self, x, y):
        beta_pilot = _fit_pilot_poisson(
            x, y, self.n_pilot_lambdas, self.pilot_position,
            self.fit_intercept, self.standardize,
        )
        self.coef_pilot_ = beta_pilot
        self.weights_ = _adaptive_weights(beta_pilot, self.eta, self.eps_pilot)
        return super().fit(x, y)


class AdaptivePoissonLassoPathCV(_AdaptivePoissonPathCVBase):
    """K-fold CV over an adaptive Poisson-lasso path."""

    _final_cls = PoissonLassoPathRegressor

    def __init__(
        self,
        *,
        eta: float = 1.0,
        eps_pilot: float = 1e-6,
        n_pilot_lambdas: int = 10,
        pilot_position: PilotPosition | int = "mid",
        cv: Any = 5,
        random_state: int | None = None,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        max_iter: int = 100,
        tol: float = 1e-6,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
    ) -> None:
        self.eta = eta
        self.eps_pilot = eps_pilot
        self.n_pilot_lambdas = n_pilot_lambdas
        self.pilot_position = pilot_position
        self.cv = cv
        self.random_state = random_state
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.max_outer = max_outer
        self.outer_tol = outer_tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.acceleration = acceleration


class AdaptivePoissonMCPPathCV(_AdaptivePoissonPathCVBase):
    """K-fold CV over an adaptive Poisson-MCP path."""

    _final_cls = PoissonMCPPathRegressor

    def __init__(
        self,
        gamma: float = 3.0,
        *,
        eta: float = 1.0,
        eps_pilot: float = 1e-6,
        n_pilot_lambdas: int = 10,
        pilot_position: PilotPosition | int = "mid",
        cv: Any = 5,
        random_state: int | None = None,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        max_iter: int = 100,
        tol: float = 1e-6,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
    ) -> None:
        self.gamma = gamma
        self.eta = eta
        self.eps_pilot = eps_pilot
        self.n_pilot_lambdas = n_pilot_lambdas
        self.pilot_position = pilot_position
        self.cv = cv
        self.random_state = random_state
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.max_outer = max_outer
        self.outer_tol = outer_tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.acceleration = acceleration

    def _extra_kwargs(self) -> dict[str, Any]:
        return {"gamma": self.gamma}


class AdaptivePoissonSCADPathCV(_AdaptivePoissonPathCVBase):
    """K-fold CV over an adaptive Poisson-SCAD path."""

    _final_cls = PoissonSCADPathRegressor

    def __init__(
        self,
        a: float = 3.7,
        *,
        eta: float = 1.0,
        eps_pilot: float = 1e-6,
        n_pilot_lambdas: int = 10,
        pilot_position: PilotPosition | int = "mid",
        cv: Any = 5,
        random_state: int | None = None,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        max_iter: int = 100,
        tol: float = 1e-6,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = False,
        acceleration: int | None = 5,
    ) -> None:
        self.a = a
        self.eta = eta
        self.eps_pilot = eps_pilot
        self.n_pilot_lambdas = n_pilot_lambdas
        self.pilot_position = pilot_position
        self.cv = cv
        self.random_state = random_state
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.max_outer = max_outer
        self.outer_tol = outer_tol
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.acceleration = acceleration

    def _extra_kwargs(self) -> dict[str, Any]:
        return {"a": self.a}


# ---- Cox (no intercept; fit(x, time, event)) ---------------------------


class _AdaptiveCoxPathBase(BaseEstimator, RegressorMixin):
    """Adaptive Cox path. `fit(x, time, event)`. No intercept (baseline
    hazard absorbs). `predict(x) = decision_function(x) = Xβ` (Cox
    prognostic index) per the existing `CoxPathRegressor` convention."""

    coefs_: NDArray[np.float64]
    lambdas_: NDArray[np.float64]
    coef_pilot_: NDArray[np.float64]
    weights_: NDArray[np.float64]
    info_: dict[str, Any]
    n_features_in_: int

    _final_cls: Any

    def _extra_kwargs(self) -> dict[str, Any]:
        return {}

    def _make_final(self, weights):
        kw: dict[str, Any] = dict(
            lambdas=self.lambdas, n_lambdas=self.n_lambdas,
            lambda_min_ratio=self.lambda_min_ratio, weights=weights,
            max_iter=self.max_iter, tol=self.tol,
            standardize=self.standardize, acceleration=self.acceleration,
            max_outer=self.max_outer, outer_tol=self.outer_tol,
        )
        kw.update(self._extra_kwargs())
        return self._final_cls(**kw)

    def fit(self, x, time, event) -> "_AdaptiveCoxPathBase":
        beta_pilot = _fit_pilot_cox(
            x, time, event, self.n_pilot_lambdas, self.pilot_position,
            self.standardize,
        )
        weights = _adaptive_weights(beta_pilot, self.eta, self.eps_pilot)
        final = self._make_final(weights).fit(x, time, event)
        self._final_estimator_ = final
        self.coefs_ = final.coefs_
        self.lambdas_ = final.lambdas_
        self.coef_pilot_ = beta_pilot
        self.weights_ = weights
        self.info_ = final.info_
        self.n_features_in_ = final.n_features_in_
        return self

    def decision_function(self, x):
        return self._final_estimator_.decision_function(x)

    def predict(self, x):
        return self._final_estimator_.predict(x)


class AdaptiveCoxLassoPathRegressor(_AdaptiveCoxPathBase):
    """Adaptive Cox lasso."""

    _final_cls = CoxMCPPathRegressor

    def __init__(
        self,
        *,
        eta: float = 1.0,
        eps_pilot: float = 1e-6,
        n_pilot_lambdas: int = 10,
        pilot_position: PilotPosition | int = "mid",
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        max_iter: int = 100,
        tol: float = 1e-6,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
        standardize: bool = False,
        acceleration: int | None = 5,
    ) -> None:
        self.eta = eta
        self.eps_pilot = eps_pilot
        self.n_pilot_lambdas = n_pilot_lambdas
        self.pilot_position = pilot_position
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.max_outer = max_outer
        self.outer_tol = outer_tol
        self.standardize = standardize
        self.acceleration = acceleration

    def _extra_kwargs(self) -> dict[str, Any]:
        return {"gamma": 1e9}


class AdaptiveCoxMCPPathRegressor(_AdaptiveCoxPathBase):
    """Adaptive Cox MCP."""

    _final_cls = CoxMCPPathRegressor

    def __init__(
        self,
        gamma: float = 3.0,
        *,
        eta: float = 1.0,
        eps_pilot: float = 1e-6,
        n_pilot_lambdas: int = 10,
        pilot_position: PilotPosition | int = "mid",
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        max_iter: int = 100,
        tol: float = 1e-6,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
        standardize: bool = False,
        acceleration: int | None = 5,
    ) -> None:
        self.gamma = gamma
        self.eta = eta
        self.eps_pilot = eps_pilot
        self.n_pilot_lambdas = n_pilot_lambdas
        self.pilot_position = pilot_position
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.max_outer = max_outer
        self.outer_tol = outer_tol
        self.standardize = standardize
        self.acceleration = acceleration

    def _extra_kwargs(self) -> dict[str, Any]:
        return {"gamma": self.gamma}


class AdaptiveCoxSCADPathRegressor(_AdaptiveCoxPathBase):
    """Adaptive Cox SCAD."""

    _final_cls = CoxSCADPathRegressor

    def __init__(
        self,
        a: float = 3.7,
        *,
        eta: float = 1.0,
        eps_pilot: float = 1e-6,
        n_pilot_lambdas: int = 10,
        pilot_position: PilotPosition | int = "mid",
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        max_iter: int = 100,
        tol: float = 1e-6,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
        standardize: bool = False,
        acceleration: int | None = 5,
    ) -> None:
        self.a = a
        self.eta = eta
        self.eps_pilot = eps_pilot
        self.n_pilot_lambdas = n_pilot_lambdas
        self.pilot_position = pilot_position
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.max_outer = max_outer
        self.outer_tol = outer_tol
        self.standardize = standardize
        self.acceleration = acceleration

    def _extra_kwargs(self) -> dict[str, Any]:
        return {"a": self.a}


class _AdaptiveCoxPathCVBase(_CoxPathCVMixin, BaseEstimator):
    """Cox adaptive CV: uses StratifiedKFold by event indicator (inherited
    from `_CoxPathCVMixin`) and Harrell c-index scoring. Pilot weights
    are computed once on full data."""

    coef_pilot_: NDArray[np.float64]
    weights_: NDArray[np.float64]
    _final_cls: Any

    def _extra_kwargs(self) -> dict[str, Any]:
        return {}

    def _make_base_path(self, **overrides):
        kw: dict[str, Any] = dict(
            lambdas=self.lambdas, n_lambdas=self.n_lambdas,
            lambda_min_ratio=self.lambda_min_ratio, weights=self.weights_,
            max_iter=self.max_iter, tol=self.tol, max_outer=self.max_outer,
            outer_tol=self.outer_tol, standardize=self.standardize,
            acceleration=self.acceleration,
        )
        kw.update(self._extra_kwargs())
        kw.update(overrides)
        return self._final_cls(**kw)

    def fit(self, x, time, event):
        beta_pilot = _fit_pilot_cox(
            x, time, event, self.n_pilot_lambdas, self.pilot_position,
            self.standardize,
        )
        self.coef_pilot_ = beta_pilot
        self.weights_ = _adaptive_weights(beta_pilot, self.eta, self.eps_pilot)
        return super().fit(x, time, event)


class AdaptiveCoxLassoPathCV(_AdaptiveCoxPathCVBase):
    """K-fold CV over an adaptive Cox-lasso path."""

    _final_cls = CoxMCPPathRegressor

    def __init__(
        self,
        *,
        eta: float = 1.0,
        eps_pilot: float = 1e-6,
        n_pilot_lambdas: int = 10,
        pilot_position: PilotPosition | int = "mid",
        cv: Any = 5,
        random_state: int | None = None,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        max_iter: int = 100,
        tol: float = 1e-6,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
        standardize: bool = False,
        acceleration: int | None = 5,
    ) -> None:
        self.eta = eta
        self.eps_pilot = eps_pilot
        self.n_pilot_lambdas = n_pilot_lambdas
        self.pilot_position = pilot_position
        self.cv = cv
        self.random_state = random_state
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.max_outer = max_outer
        self.outer_tol = outer_tol
        self.standardize = standardize
        self.acceleration = acceleration

    def _extra_kwargs(self) -> dict[str, Any]:
        return {"gamma": 1e9}


class AdaptiveCoxMCPPathCV(_AdaptiveCoxPathCVBase):
    """K-fold CV over an adaptive Cox-MCP path."""

    _final_cls = CoxMCPPathRegressor

    def __init__(
        self,
        gamma: float = 3.0,
        *,
        eta: float = 1.0,
        eps_pilot: float = 1e-6,
        n_pilot_lambdas: int = 10,
        pilot_position: PilotPosition | int = "mid",
        cv: Any = 5,
        random_state: int | None = None,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        max_iter: int = 100,
        tol: float = 1e-6,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
        standardize: bool = False,
        acceleration: int | None = 5,
    ) -> None:
        self.gamma = gamma
        self.eta = eta
        self.eps_pilot = eps_pilot
        self.n_pilot_lambdas = n_pilot_lambdas
        self.pilot_position = pilot_position
        self.cv = cv
        self.random_state = random_state
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.max_outer = max_outer
        self.outer_tol = outer_tol
        self.standardize = standardize
        self.acceleration = acceleration

    def _extra_kwargs(self) -> dict[str, Any]:
        return {"gamma": self.gamma}


class AdaptiveCoxSCADPathCV(_AdaptiveCoxPathCVBase):
    """K-fold CV over an adaptive Cox-SCAD path."""

    _final_cls = CoxSCADPathRegressor

    def __init__(
        self,
        a: float = 3.7,
        *,
        eta: float = 1.0,
        eps_pilot: float = 1e-6,
        n_pilot_lambdas: int = 10,
        pilot_position: PilotPosition | int = "mid",
        cv: Any = 5,
        random_state: int | None = None,
        lambdas: NDArray[np.float64] | None = None,
        n_lambdas: int = 100,
        lambda_min_ratio: float = 1e-3,
        max_iter: int = 100,
        tol: float = 1e-6,
        max_outer: int = 10,
        outer_tol: float = 1e-6,
        standardize: bool = False,
        acceleration: int | None = 5,
    ) -> None:
        self.a = a
        self.eta = eta
        self.eps_pilot = eps_pilot
        self.n_pilot_lambdas = n_pilot_lambdas
        self.pilot_position = pilot_position
        self.cv = cv
        self.random_state = random_state
        self.lambdas = lambdas
        self.n_lambdas = n_lambdas
        self.lambda_min_ratio = lambda_min_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.max_outer = max_outer
        self.outer_tol = outer_tol
        self.standardize = standardize
        self.acceleration = acceleration

    def _extra_kwargs(self) -> dict[str, Any]:
        return {"a": self.a}


__all__ = [
    "AdaptiveLassoPathRegressor",
    "AdaptiveMCPPathRegressor",
    "AdaptiveSCADPathRegressor",
    "AdaptiveLassoPathCV",
    "AdaptiveMCPPathCV",
    "AdaptiveSCADPathCV",
    "AdaptiveGroupLassoPathRegressor",
    "AdaptiveGroupMCPPathRegressor",
    "AdaptiveGroupSCADPathRegressor",
    "AdaptiveGroupLassoPathCV",
    "AdaptiveGroupMCPPathCV",
    "AdaptiveGroupSCADPathCV",
    "AdaptiveLogisticLassoPathRegressor",
    "AdaptiveLogisticMCPPathRegressor",
    "AdaptiveLogisticSCADPathRegressor",
    "AdaptiveLogisticLassoPathCV",
    "AdaptiveLogisticMCPPathCV",
    "AdaptiveLogisticSCADPathCV",
    "AdaptivePoissonLassoPathRegressor",
    "AdaptivePoissonMCPPathRegressor",
    "AdaptivePoissonSCADPathRegressor",
    "AdaptivePoissonLassoPathCV",
    "AdaptivePoissonMCPPathCV",
    "AdaptivePoissonSCADPathCV",
    "AdaptiveCoxLassoPathRegressor",
    "AdaptiveCoxMCPPathRegressor",
    "AdaptiveCoxSCADPathRegressor",
    "AdaptiveCoxLassoPathCV",
    "AdaptiveCoxMCPPathCV",
    "AdaptiveCoxSCADPathCV",
]
