"""Debiased / desparsified lasso for least squares.

Van de Geer–Bühlmann–Ritov–Dezeure (2014) confidence intervals and
p-values for high-dimensional lasso regression. Constructs an
approximate inverse Gram `Θ̂` via per-column **nodewise lassos**
(Meinshausen–Bühlmann 2006), uses it to debias the penalized
estimator, and reports asymptotic Gaussian inference on every
coordinate of `β̂_d`.

The debiased estimator is

    β̂_d = β̂ + (1 / n) · Θ̂ · Xᵀ (y − X β̂)

with asymptotic distribution (under standard sparsity / restricted-
eigenvalue conditions)

    √n · (β̂_d − β) ⇝ N(0, σ² · Θ̂ Σ̂ Θ̂ᵀ).

This is the one inference feature `glmnet` and `ncvreg` do not
offer; the canonical R implementation is `hdi::lasso.proj`. We
match its semantics: free function on `(X, y)` returning a result
dataclass plus a thin sklearn-style `DebiasedLassoRegressor`
wrapper.

Scope (v1): least-squares only, dense `X`. GLM (logistic / Poisson)
debiasing via the weighted-LS surrogate + Fisher information is a
planned follow-up.

References
----------
- Van de Geer, S., Bühlmann, P., Ritov, Y., Dezeure, R. (2014).
  *On asymptotically optimal confidence regions and tests for
  high-dimensional models.* Annals of Statistics 42(3): 1166–1202.
- Zhang, C.-H., Zhang, S. (2014). *Confidence intervals for low-
  dimensional parameters in high-dimensional linear models.*
  JRSS B 76(1): 217–242.
- Reid, S., Tibshirani, R., Friedman, J. (2016). *A study of error
  variance estimation in lasso regression.* Statistica Sinica 26:
  35–67.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from sklearn.base import BaseEstimator, RegressorMixin

from skein_glm.estimators import ElasticNetRegressor

_ACTIVE_EPS = 1e-10


# --- result dataclass -------------------------------------------------


@dataclass
class DebiasedLassoResult:
    """Output of :func:`debiased_lasso`.

    All coefficient arrays are returned on the **original (un-
    standardized) feature scale**. ``Theta`` is on the standardized
    scale used internally (where the nodewise lasso defaults are
    dimensionless); it is exposed for inspection and reuse.

    Attributes
    ----------
    coef_debiased : ndarray (p,)
        Desparsified estimator `β̂_d`.
    coef_lasso : ndarray (p,)
        The underlying penalized lasso fit `β̂` (also original-scale).
    intercept_ : float
    se : ndarray (p,)
        Asymptotic standard error from the diagonal of
        `σ̂² · Θ̂ Σ̂ Θ̂ᵀ / n`.
    ci_lower, ci_upper : ndarray (p,)
        Two-sided `(1 − alpha)` CIs.
    pvalues : ndarray (p,)
        Two-sided Wald p-values for `H_0: β_j = 0`.
    z_scores : ndarray (p,)
        `β̂_d / se`.
    sigma_hat : float
        Residual scale `‖y − ŷ‖ / sqrt(n − ‖β̂‖₀)`.
    Theta : ndarray (p, p)
        Approximate inverse Gram on the standardized scale.
    lambda_main : float
    lambda_nodewise : ndarray (p,)
    alpha : float
        CI level used.
    """

    coef_debiased: NDArray[np.float64]
    coef_lasso: NDArray[np.float64]
    intercept_: float
    se: NDArray[np.float64]
    ci_lower: NDArray[np.float64]
    ci_upper: NDArray[np.float64]
    pvalues: NDArray[np.float64]
    z_scores: NDArray[np.float64]
    sigma_hat: float
    Theta: NDArray[np.float64]
    lambda_main: float
    lambda_nodewise: NDArray[np.float64]
    alpha: float


# --- helpers ----------------------------------------------------------


def _theoretical_lambda(n: int, p: int) -> float:
    """VBR theoretical scale `√(2 log p / n)`.

    Dimensionless — appropriate when columns are standardized. For
    the main lasso this is the *base* scale; users typically prefer
    a CV-tuned `lambda_` and override the default.
    """
    return float(np.sqrt(2.0 * np.log(max(p, 2)) / n))


def _fit_nodewise_column(
    Xs: NDArray[np.float64],
    j: int,
    lambda_j: float,
    max_iter: int,
    tol: float,
) -> tuple[NDArray[np.float64], float]:
    """Run the j-th nodewise lasso: regress ``X_s[:, j]`` on the
    remaining columns.

    Returns
    -------
    gamma : ndarray (p - 1,)
        Lasso coefficients with ``j`` deleted.
    tau2 : float
        VBR `τ̂_j² = ‖X_j − X_{−j} γ̂‖² / n + λ_j · ‖γ̂‖₁`. The `+ λ‖γ‖₁`
        correction is the residual-norm-plus-penalty term that makes
        Θ̂ exactly satisfy the row-wise KKT inversion identity (VBR
        Lemma 6.2). Using the plain residual norm gives a slightly
        biased Θ̂ at finite `n`.
    """
    n = Xs.shape[0]
    Xmj = np.delete(Xs, j, axis=1)
    xj = Xs[:, j].copy()

    fit = ElasticNetRegressor(
        lambda_=lambda_j,
        alpha=1.0,
        fit_intercept=False,
        standardize=False,
        max_iter=max_iter,
        tol=tol,
    ).fit(Xmj, xj)
    gamma = np.asarray(fit.coef_, dtype=np.float64)

    resid = xj - Xmj @ gamma
    tau2 = float(np.dot(resid, resid)) / n + lambda_j * float(np.sum(np.abs(gamma)))
    return gamma, tau2


def _assemble_theta_rows(
    gammas: list[NDArray[np.float64]],
    tau2s: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Build Θ̂ row-by-row from the nodewise outputs.

    Row j is `(−γ̂_{j,1}, …, 1, …, −γ̂_{j,p−1}) / τ̂_j²` with the `1`
    in position j. Θ̂ is **not symmetrized** here; VBR's variance
    formula uses the symmetric quadratic form `Θ̂ Σ̂ Θ̂ᵀ` so
    symmetrization at this stage is unnecessary.
    """
    p = len(gammas)
    Theta = np.zeros((p, p), dtype=np.float64)
    for j, (g, t2) in enumerate(zip(gammas, tau2s)):
        row = np.empty(p, dtype=np.float64)
        row[:j] = -g[:j]
        row[j] = 1.0
        row[j + 1 :] = -g[j:]
        Theta[j, :] = row / t2
    return Theta


# --- main entry point ------------------------------------------------


def debiased_lasso(
    X: Any,
    y: Any,
    *,
    lambda_: float | None = None,
    lambda_nodewise: float | NDArray[np.float64] | None = None,
    alpha: float = 0.05,
    fit_intercept: bool = True,
    standardize: bool = True,
    max_iter: int = 1000,
    tol: float = 1e-7,
    n_jobs: int | None = None,
) -> DebiasedLassoResult:
    """Van de Geer–Bühlmann–Ritov debiased lasso for least squares.

    Parameters
    ----------
    X : array-like (n, p)
    y : array-like (n,)
    lambda_ : float or None, default None
        Regularisation for the **main** lasso fit. If ``None``, the
        theoretical scale `√(2 log p / n)` is used on standardized
        features. For best inference quality, pass a CV-tuned λ from
        a prior :class:`~skein_glm.ElasticNetPathCV` fit.
    lambda_nodewise : float, array-like (p,), or None, default None
        Per-column λ for the **nodewise** lassos that build `Θ̂`.
        Scalar broadcasts to every column; an array selects per
        column. ``None`` uses `√(2 log p / n)` uniformly (the VBR
        theoretical choice on standardized columns).
    alpha : float, default 0.05
        Two-sided CI / p-value level.
    fit_intercept : bool, default True
        Center `y` and `X`; intercept is recovered post-fit. The
        intercept itself is **not** an inferential target — VBR
        theory applies to the slopes.
    standardize : bool, default True
        Scale columns of `X` to unit variance before fitting. The
        dimensionless `lambda_nodewise` default is calibrated to
        standardized columns; turning this off without supplying
        column-specific `lambda_nodewise` typically gives wrong
        scale on the CIs.
    max_iter, tol : int, float
        Forwarded to every underlying lasso fit.
    n_jobs : int or None
        joblib parallelism over the `p` nodewise lassos. ``-1``
        uses all cores. The main lasso is single-threaded; the
        nodewise loop dominates for `p ≳ 50`.

    Returns
    -------
    DebiasedLassoResult

    Notes
    -----
    Memory is `O(p²)` for `Θ̂` plus `O(n p)` for the standardized
    design — practical up to `p ~ 5000` on a modern laptop. For
    larger `p`, the nodewise loop runtime (`p` lasso fits) is the
    binding constraint rather than memory.

    The asymptotic distribution requires (a) lasso consistency at
    rate `‖β̂ − β‖₁ = o(1/√(log p))`, (b) row-wise sparsity of the
    true `Σ⁻¹`, (c) sub-Gaussian design and noise. Failures show up
    as poor coverage in the empirical CI simulation tests.
    """
    from joblib import Parallel, delayed

    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError(
            f"y must be 1D with length {X.shape[0]}, got shape {y.shape}"
        )
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be ≥ 1; got {max_iter}")
    if tol <= 0:
        raise ValueError(f"tol must be > 0; got {tol}")

    n, p = X.shape
    if p < 2:
        raise ValueError(
            f"debiased lasso requires p ≥ 2 features; got p = {p}"
        )

    # Center / scale.
    if fit_intercept:
        x_mean = X.mean(axis=0)
        y_mean = float(y.mean())
    else:
        x_mean = np.zeros(p)
        y_mean = 0.0
    Xc = X - x_mean
    yc = y - y_mean

    if standardize:
        x_scale = Xc.std(axis=0, ddof=0)
        # Constant column ⇒ keep scale 1 so we don't divide by zero;
        # the lasso will still drop it cleanly via the standard KKT.
        x_scale = np.where(x_scale > 0, x_scale, 1.0)
    else:
        x_scale = np.ones(p)
    Xs = Xc / x_scale

    # Main lasso fit (standardized scale).
    lambda_main = (
        _theoretical_lambda(n, p) if lambda_ is None else float(lambda_)
    )
    if lambda_main <= 0:
        raise ValueError(f"lambda_ must be > 0; got {lambda_main}")
    main_fit = ElasticNetRegressor(
        lambda_=lambda_main,
        alpha=1.0,
        fit_intercept=False,
        standardize=False,
        max_iter=max_iter,
        tol=tol,
    ).fit(Xs, yc)
    beta_hat_s = np.asarray(main_fit.coef_, dtype=np.float64)

    # Nodewise lassos.
    if lambda_nodewise is None:
        lam_nw = np.full(p, _theoretical_lambda(n, p), dtype=np.float64)
    elif np.isscalar(lambda_nodewise):
        lam_nw = np.full(
            p, float(lambda_nodewise),  # type: ignore[arg-type]
            dtype=np.float64,
        )
    else:
        lam_nw = np.ascontiguousarray(lambda_nodewise, dtype=np.float64)
        if lam_nw.shape != (p,):
            raise ValueError(
                f"lambda_nodewise must be scalar or shape ({p},); "
                f"got {lam_nw.shape}"
            )
    if np.any(lam_nw <= 0):
        raise ValueError("lambda_nodewise entries must be > 0")

    results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_nodewise_column)(Xs, j, float(lam_nw[j]), max_iter, tol)
        for j in range(p)
    )
    gammas = [g for g, _ in results]
    tau2s = np.array([t for _, t in results], dtype=np.float64)
    Theta = _assemble_theta_rows(gammas, tau2s)  # (p, p), standardized scale

    # Debiased estimator (standardized scale).
    resid_s = yc - Xs @ beta_hat_s
    beta_d_s = beta_hat_s + (Theta @ (Xs.T @ resid_s)) / n

    # σ̂ from lasso residuals — Reid–Tibshirani–Friedman convention.
    nz = int(np.sum(np.abs(beta_hat_s) > _ACTIVE_EPS))
    sigma2 = float(np.dot(resid_s, resid_s)) / max(n - nz, 1)
    sigma_hat = float(np.sqrt(sigma2))

    # Variance: σ̂² · diag(Θ̂ Σ̂ Θ̂ᵀ) / n.
    # Compute via U = X_s · Θ̂ᵀ ∈ R^{n×p}; then [Θ̂ Σ̂ Θ̂ᵀ]_jj = ‖U_j‖² / n.
    # That avoids materializing the p×p Σ̂.
    U = Xs @ Theta.T
    diag_quad = np.einsum("ij,ij->j", U, U) / n
    var_d_s = sigma2 * diag_quad / n
    se_s = np.sqrt(np.maximum(var_d_s, 0.0))

    # Back to original scale.
    beta_d = beta_d_s / x_scale
    beta_lasso = beta_hat_s / x_scale
    se = se_s / x_scale

    intercept = (
        y_mean - float(np.dot(beta_d, x_mean)) if fit_intercept else 0.0
    )

    # Inference (Wald).
    z_alpha = float(stats.norm.ppf(1.0 - alpha / 2.0))
    safe_se = np.where(se > 0, se, 1.0)
    z_scores = np.where(se > 0, beta_d / safe_se, 0.0)
    pvalues = 2.0 * stats.norm.sf(np.abs(z_scores))
    ci_lower = beta_d - z_alpha * se
    ci_upper = beta_d + z_alpha * se

    return DebiasedLassoResult(
        coef_debiased=beta_d,
        coef_lasso=beta_lasso,
        intercept_=intercept,
        se=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        pvalues=pvalues,
        z_scores=z_scores,
        sigma_hat=sigma_hat,
        Theta=Theta,
        lambda_main=lambda_main,
        lambda_nodewise=lam_nw,
        alpha=alpha,
    )


# --- sklearn-style wrapper -------------------------------------------


class DebiasedLassoRegressor(BaseEstimator, RegressorMixin):
    """Sklearn-style facade over :func:`debiased_lasso`.

    Exposes the debiased estimator as ``coef_`` / ``intercept_`` so
    the result composes with sklearn `Pipeline` and metric helpers.
    The VBR-specific attributes (``se_``, ``ci_lower_``,
    ``ci_upper_``, ``pvalues_``, ``z_scores_``, ``Theta_``,
    ``sigma_hat_``) live on the fitted estimator.

    Parameters mirror :func:`debiased_lasso`.

    Examples
    --------
    >>> from skein_glm import DebiasedLassoRegressor
    >>> est = DebiasedLassoRegressor(random_state=0).fit(X, y)
    >>> est.coef_, est.ci_lower_, est.ci_upper_, est.pvalues_
    """

    coef_: NDArray[np.float64]
    coef_lasso_: NDArray[np.float64]
    intercept_: float
    se_: NDArray[np.float64]
    ci_lower_: NDArray[np.float64]
    ci_upper_: NDArray[np.float64]
    pvalues_: NDArray[np.float64]
    z_scores_: NDArray[np.float64]
    sigma_hat_: float
    Theta_: NDArray[np.float64]
    lambda_main_: float
    lambda_nodewise_: NDArray[np.float64]
    n_features_in_: int

    def __init__(
        self,
        *,
        lambda_: float | None = None,
        lambda_nodewise: float | NDArray[np.float64] | None = None,
        alpha: float = 0.05,
        fit_intercept: bool = True,
        standardize: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-7,
        n_jobs: int | None = None,
    ) -> None:
        self.lambda_ = lambda_
        self.lambda_nodewise = lambda_nodewise
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.max_iter = max_iter
        self.tol = tol
        self.n_jobs = n_jobs

    def fit(self, X: Any, y: Any) -> "DebiasedLassoRegressor":
        res = debiased_lasso(
            X, y,
            lambda_=self.lambda_,
            lambda_nodewise=self.lambda_nodewise,
            alpha=self.alpha,
            fit_intercept=self.fit_intercept,
            standardize=self.standardize,
            max_iter=self.max_iter,
            tol=self.tol,
            n_jobs=self.n_jobs,
        )
        self.coef_ = res.coef_debiased
        self.coef_lasso_ = res.coef_lasso
        self.intercept_ = res.intercept_
        self.se_ = res.se
        self.ci_lower_ = res.ci_lower
        self.ci_upper_ = res.ci_upper
        self.pvalues_ = res.pvalues
        self.z_scores_ = res.z_scores
        self.sigma_hat_ = res.sigma_hat
        self.Theta_ = res.Theta
        self.lambda_main_ = res.lambda_main
        self.lambda_nodewise_ = res.lambda_nodewise
        self.n_features_in_ = int(res.coef_debiased.shape[0])
        return self

    def predict(self, X: Any) -> NDArray[np.float64]:
        X = np.ascontiguousarray(X, dtype=np.float64)
        return X @ self.coef_ + self.intercept_


__all__ = [
    "DebiasedLassoResult",
    "debiased_lasso",
    "DebiasedLassoRegressor",
]
