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

from skein_glm.estimators import (
    CoxMCPRegressor,
    ElasticNetRegressor,
    LogisticLassoRegressor,
    PoissonLassoRegressor,
)

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


# =====================================================================
# M5.x-b — Debiased GLM (logistic + Poisson)
# =====================================================================
#
# Extends the VBR construction from LS to canonical-link GLMs via the
# weighted-LS surrogate. The estimating equation at β̂ is
#
#     score(β̂) = Xᵀ (y − μ̂(β̂))
#
# and the Fisher information at β̂ is J = (1/n) · Xᵀ W X where W is the
# diagonal of working weights — μ̂(1−μ̂) for binomial logit, μ̂ for
# Poisson log. The debiased estimator is
#
#     β̂_d = β̂ + (1/n) · Θ̂ · Xᵀ (y − μ̂)
#
# with asymptotic distribution √n (β̂_d − β) ⇝ N(0, J⁻¹) and Θ̂ ≈ J⁻¹
# constructed nodewise on the **weighted design** X̃ = W^{1/2} X. This
# is the JM-extended-to-GLM construction (Van de Geer 2014 §3).
#
# Reuses :func:`_fit_nodewise_column` and :func:`_assemble_theta_rows`
# from the LS path — the math is identical once you swap `X` for `X̃`.


_GLM_WEIGHT_FLOOR = 1e-8


@dataclass
class DebiasedGLMResult:
    """Output of :func:`debiased_logistic_lasso` / :func:`debiased_poisson_lasso`.

    Same fields as :class:`DebiasedLassoResult` minus ``sigma_hat``
    (GLMs have no residual scale — the noise is encoded in the working
    weights `W`) and plus ``mu_fitted`` and ``family``.

    Attributes
    ----------
    coef_debiased : ndarray (p,)
    coef_glm : ndarray (p,)
        Underlying penalized GLM fit on the original feature scale.
    intercept_ : float
        Intercept of the underlying fit; not an inferential target (VBR
        theory applies to slopes only).
    se : ndarray (p,)
        `sqrt(diag(Θ̂ X̃ᵀX̃ Θ̂ᵀ)) / n`.
    ci_lower, ci_upper, pvalues, z_scores : ndarray (p,)
    mu_fitted : ndarray (n,)
        `μ̂` at the penalized fit. Useful for diagnostics.
    Theta : ndarray (p, p)
        Approximate inverse Fisher on the standardized scale.
    lambda_main : float
    lambda_nodewise : ndarray (p,)
    alpha : float
    family : str
        ``"binomial"`` or ``"poisson"``.
    """

    coef_debiased: NDArray[np.float64]
    coef_glm: NDArray[np.float64]
    intercept_: float
    se: NDArray[np.float64]
    ci_lower: NDArray[np.float64]
    ci_upper: NDArray[np.float64]
    pvalues: NDArray[np.float64]
    z_scores: NDArray[np.float64]
    mu_fitted: NDArray[np.float64]
    Theta: NDArray[np.float64]
    lambda_main: float
    lambda_nodewise: NDArray[np.float64]
    alpha: float
    family: str


def _debiased_glm_core(
    Xs: NDArray[np.float64],
    y: NDArray[np.float64],
    beta_hat_s: NDArray[np.float64],
    intercept_s: float,
    mu_hat: NDArray[np.float64],
    w_diag: NDArray[np.float64],
    x_mean: NDArray[np.float64],
    x_scale: NDArray[np.float64],
    lam_nw: NDArray[np.float64],
    alpha: float,
    max_iter: int,
    tol: float,
    n_jobs: int | None,
    family: str,
    lambda_main: float,
) -> DebiasedGLMResult:
    """Shared GLM debiasing core — nodewise + variance + inference.

    Inputs are on the **standardized** feature scale (mean-centered,
    scaled to unit variance). Output is destandardized to the original
    scale.
    """
    from joblib import Parallel, delayed

    n, p = Xs.shape

    # Floor working weights to avoid degenerate columns in X̃ near
    # saturation (logistic prob ≈ 0/1) or very small Poisson means.
    # The asymptotic theory requires the working weights to be bounded
    # away from zero; in practice tiny floors here don't move the
    # answer materially when the data are well-conditioned but prevent
    # blow-ups when they aren't.
    w_safe = np.maximum(w_diag, _GLM_WEIGHT_FLOOR)
    w_sqrt = np.sqrt(w_safe)
    X_tilde = Xs * w_sqrt[:, None]

    # Nodewise lassos on the weighted design.
    results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_nodewise_column)(X_tilde, j, float(lam_nw[j]), max_iter, tol)
        for j in range(p)
    )
    gammas = [g for g, _ in results]
    tau2s = np.array([t for _, t in results], dtype=np.float64)
    Theta = _assemble_theta_rows(gammas, tau2s)

    # Debias: β̂_d = β̂ + (1/n) Θ̂ Xᵀ (y − μ̂). Note this is the
    # *unweighted* score (the canonical-link gradient), not the
    # weighted X̃ᵀ (y − μ̂).
    score = Xs.T @ (y - mu_hat)
    beta_d_s = beta_hat_s + (Theta @ score) / n

    # Variance: diag(Θ̂ X̃ᵀX̃ Θ̂ᵀ) / n. Compute via U = X̃ Θ̂ᵀ to skip
    # the (p, p) Gram materialization.
    U = X_tilde @ Theta.T
    var_d_s = np.einsum("ij,ij->j", U, U) / (n * n)
    se_s = np.sqrt(np.maximum(var_d_s, 0.0))

    # Destandardize to original feature scale.
    beta_d = beta_d_s / x_scale
    beta_glm = beta_hat_s / x_scale
    se = se_s / x_scale
    intercept = intercept_s - float(np.dot(beta_d, x_mean))

    # Wald inference.
    z_alpha = float(stats.norm.ppf(1.0 - alpha / 2.0))
    safe_se = np.where(se > 0, se, 1.0)
    z_scores = np.where(se > 0, beta_d / safe_se, 0.0)
    pvalues = 2.0 * stats.norm.sf(np.abs(z_scores))
    ci_lower = beta_d - z_alpha * se
    ci_upper = beta_d + z_alpha * se

    return DebiasedGLMResult(
        coef_debiased=beta_d,
        coef_glm=beta_glm,
        intercept_=intercept,
        se=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        pvalues=pvalues,
        z_scores=z_scores,
        mu_fitted=mu_hat,
        Theta=Theta,
        lambda_main=lambda_main,
        lambda_nodewise=lam_nw,
        alpha=alpha,
        family=family,
    )


def _standardize_X(
    X: NDArray[np.float64],
    standardize: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Mean-center and (optionally) unit-scale columns.

    Returns ``(Xs, x_mean, x_scale)``. When ``standardize=False`` we
    still center; ``x_scale`` is all ones. Constant columns get
    ``scale=1`` so we don't divide by zero — the penalized fit will
    drop them via the usual KKT.
    """
    x_mean = X.mean(axis=0)
    Xc = X - x_mean
    if standardize:
        x_scale = Xc.std(axis=0, ddof=0)
        x_scale = np.where(x_scale > 0, x_scale, 1.0)
    else:
        x_scale = np.ones(X.shape[1], dtype=np.float64)
    return Xc / x_scale, x_mean, x_scale


def _build_lambda_nodewise(
    lambda_nodewise: float | NDArray[np.float64] | None,
    n: int,
    p: int,
) -> NDArray[np.float64]:
    if lambda_nodewise is None:
        return np.full(p, _theoretical_lambda(n, p), dtype=np.float64)
    if np.isscalar(lambda_nodewise):
        return np.full(
            p, float(lambda_nodewise),  # type: ignore[arg-type]
            dtype=np.float64,
        )
    arr = np.ascontiguousarray(lambda_nodewise, dtype=np.float64)
    if arr.shape != (p,):
        raise ValueError(
            f"lambda_nodewise must be scalar or shape ({p},); got {arr.shape}"
        )
    return arr


def _sigmoid(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Numerically stable sigmoid."""
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


# --- logistic --------------------------------------------------------


def debiased_logistic_lasso(
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
) -> DebiasedGLMResult:
    """VBR debiased lasso for binomial logistic regression.

    Construction (Van de Geer 2014 §3): fit penalized logistic lasso
    to obtain ``β̂``, compute ``μ̂_i = σ(η̂_i)`` and working weights
    ``W_ii = μ̂_i (1 − μ̂_i)``, then build the approximate inverse
    Fisher ``Θ̂`` by nodewise lassos on ``X̃ = W^{1/2} X``. The
    debiased estimator is
    ``β̂_d = β̂ + (1/n) · Θ̂ · Xᵀ (y − μ̂)``
    with asymptotic variance ``diag(Θ̂ X̃ᵀX̃ Θ̂ᵀ) / n`` (no `σ²` —
    the noise is encoded in `W`).

    Parameters mirror :func:`debiased_lasso`. The penalized fit is
    obtained from :class:`~skein_glm.LogisticLassoRegressor`, the new
    convex-L1 primitive — debiasing on top of the prior
    ``MCP(γ=1e9)`` approximation would have inherited its bias.

    Notes
    -----
    Working weights are floored at ``1e-8`` to avoid degenerate
    columns in `X̃` when fitted probabilities saturate near 0 or 1.
    If you see suspiciously narrow CIs, inspect ``mu_fitted`` for
    boundary cases.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError(
            f"y must be 1D with length {X.shape[0]}, got shape {y.shape}"
        )
    if not np.all((y == 0.0) | (y == 1.0)):
        raise ValueError("logistic regression requires y ∈ {0, 1}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    n, p = X.shape
    if p < 2:
        raise ValueError(f"debiased lasso requires p ≥ 2 features; got p = {p}")

    Xs, x_mean, x_scale = _standardize_X(X, standardize)
    # The logistic primitive fits an intercept itself; we pass
    # standardized X. (`fit_intercept` here means: do we let the GLM
    # fit one at all.)
    lambda_main = (
        _theoretical_lambda(n, p) if lambda_ is None else float(lambda_)
    )
    if lambda_main <= 0:
        raise ValueError(f"lambda_ must be > 0; got {lambda_main}")
    fit = LogisticLassoRegressor(
        lambda_=lambda_main,
        fit_intercept=fit_intercept,
        standardize=False,
        max_iter=max_iter,
        tol=tol,
        max_outer=20,
        outer_tol=tol,
    ).fit(Xs, y)
    beta_hat_s = np.asarray(fit.coef_, dtype=np.float64)
    intercept_s = float(fit.intercept_)

    eta = Xs @ beta_hat_s + intercept_s
    mu_hat = _sigmoid(eta)
    w_diag = mu_hat * (1.0 - mu_hat)

    lam_nw = _build_lambda_nodewise(lambda_nodewise, n, p)
    if np.any(lam_nw <= 0):
        raise ValueError("lambda_nodewise entries must be > 0")

    return _debiased_glm_core(
        Xs=Xs, y=y, beta_hat_s=beta_hat_s, intercept_s=intercept_s,
        mu_hat=mu_hat, w_diag=w_diag,
        x_mean=x_mean, x_scale=x_scale,
        lam_nw=lam_nw, alpha=alpha,
        max_iter=max_iter, tol=tol, n_jobs=n_jobs,
        family="binomial", lambda_main=lambda_main,
    )


# --- Poisson --------------------------------------------------------


def debiased_poisson_lasso(
    X: Any,
    y: Any,
    *,
    lambda_: float | None = None,
    lambda_nodewise: float | NDArray[np.float64] | None = None,
    alpha: float = 0.05,
    fit_intercept: bool = True,
    standardize: bool = True,
    offset: NDArray[np.float64] | None = None,
    max_iter: int = 1000,
    tol: float = 1e-7,
    n_jobs: int | None = None,
) -> DebiasedGLMResult:
    """VBR debiased lasso for Poisson log-link regression.

    Construction: fit penalized Poisson lasso to obtain ``β̂``,
    compute ``μ̂_i = exp(η̂_i + offset_i)`` and working weights
    ``W_ii = μ̂_i``, build ``Θ̂`` nodewise on ``X̃ = W^{1/2} X``,
    debias as in the logistic case.

    Parameters mirror :func:`debiased_logistic_lasso` plus a Poisson
    ``offset`` (log-exposure) that participates in the linear
    predictor but is not an inferential target.

    Notes
    -----
    Poisson working weights can have very large dynamic range
    (`μ̂ = exp(η̂)`). If some `μ̂_i` are tiny, the corresponding rows
    of `X̃` are heavily down-weighted and the asymptotic theory's
    regularity conditions may be strained. The same `1e-8` floor as
    the logistic case applies — well-conditioned problems are
    unaffected.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError(
            f"y must be 1D with length {X.shape[0]}, got shape {y.shape}"
        )
    if not np.all(np.isfinite(y)) or np.any(y < 0.0):
        raise ValueError("Poisson regression requires y ≥ 0 (finite)")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    n, p = X.shape
    if p < 2:
        raise ValueError(f"debiased lasso requires p ≥ 2 features; got p = {p}")
    offset_arr: NDArray[np.float64] | None = None
    if offset is not None:
        offset_arr = np.ascontiguousarray(offset, dtype=np.float64)
        if offset_arr.shape != (n,):
            raise ValueError(
                f"offset must be 1D with length {n}; got shape {offset_arr.shape}"
            )

    Xs, x_mean, x_scale = _standardize_X(X, standardize)
    lambda_main = (
        _theoretical_lambda(n, p) if lambda_ is None else float(lambda_)
    )
    if lambda_main <= 0:
        raise ValueError(f"lambda_ must be > 0; got {lambda_main}")

    fit_kwargs: dict[str, Any] = dict(
        lambda_=lambda_main,
        fit_intercept=fit_intercept,
        standardize=False,
        max_iter=max_iter,
        tol=tol,
        max_outer=20,
        outer_tol=tol,
    )
    if offset_arr is not None:
        fit_kwargs["offset"] = offset_arr
    fit = PoissonLassoRegressor(**fit_kwargs).fit(Xs, y)
    beta_hat_s = np.asarray(fit.coef_, dtype=np.float64)
    intercept_s = float(fit.intercept_)

    eta = Xs @ beta_hat_s + intercept_s
    if offset_arr is not None:
        eta = eta + offset_arr
    mu_hat = np.exp(eta)
    w_diag = mu_hat

    lam_nw = _build_lambda_nodewise(lambda_nodewise, n, p)
    if np.any(lam_nw <= 0):
        raise ValueError("lambda_nodewise entries must be > 0")

    return _debiased_glm_core(
        Xs=Xs, y=y, beta_hat_s=beta_hat_s, intercept_s=intercept_s,
        mu_hat=mu_hat, w_diag=w_diag,
        x_mean=x_mean, x_scale=x_scale,
        lam_nw=lam_nw, alpha=alpha,
        max_iter=max_iter, tol=tol, n_jobs=n_jobs,
        family="poisson", lambda_main=lambda_main,
    )


# --- sklearn-style wrappers -----------------------------------------


class _DebiasedGLMRegressorBase(BaseEstimator):
    """Shared facade: exposes ``coef_`` / ``intercept_`` and the
    inference outputs as suffixed attributes. Subclasses pin the
    family-specific free function."""

    coef_: NDArray[np.float64]
    coef_glm_: NDArray[np.float64]
    intercept_: float
    se_: NDArray[np.float64]
    ci_lower_: NDArray[np.float64]
    ci_upper_: NDArray[np.float64]
    pvalues_: NDArray[np.float64]
    z_scores_: NDArray[np.float64]
    Theta_: NDArray[np.float64]
    mu_fitted_: NDArray[np.float64]
    lambda_main_: float
    lambda_nodewise_: NDArray[np.float64]
    family_: str
    n_features_in_: int

    def _store_result(self, res: DebiasedGLMResult) -> None:
        self.coef_ = res.coef_debiased
        self.coef_glm_ = res.coef_glm
        self.intercept_ = res.intercept_
        self.se_ = res.se
        self.ci_lower_ = res.ci_lower
        self.ci_upper_ = res.ci_upper
        self.pvalues_ = res.pvalues
        self.z_scores_ = res.z_scores
        self.Theta_ = res.Theta
        self.mu_fitted_ = res.mu_fitted
        self.lambda_main_ = res.lambda_main
        self.lambda_nodewise_ = res.lambda_nodewise
        self.family_ = res.family
        self.n_features_in_ = int(res.coef_debiased.shape[0])

    def decision_function(self, X: Any) -> NDArray[np.float64]:
        X = np.ascontiguousarray(X, dtype=np.float64)
        return X @ self.coef_ + self.intercept_


class DebiasedLogisticLassoRegressor(_DebiasedGLMRegressorBase):
    """Sklearn-style facade over :func:`debiased_logistic_lasso`."""

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

    def fit(self, X: Any, y: Any) -> "DebiasedLogisticLassoRegressor":
        res = debiased_logistic_lasso(
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
        self._store_result(res)
        return self

    def predict_proba(self, X: Any) -> NDArray[np.float64]:
        return _sigmoid(self.decision_function(X))

    def predict(self, X: Any) -> NDArray[np.float64]:
        return (self.predict_proba(X) >= 0.5).astype(np.float64)


class DebiasedPoissonLassoRegressor(_DebiasedGLMRegressorBase):
    """Sklearn-style facade over :func:`debiased_poisson_lasso`."""

    def __init__(
        self,
        *,
        lambda_: float | None = None,
        lambda_nodewise: float | NDArray[np.float64] | None = None,
        alpha: float = 0.05,
        fit_intercept: bool = True,
        standardize: bool = True,
        offset: NDArray[np.float64] | None = None,
        max_iter: int = 1000,
        tol: float = 1e-7,
        n_jobs: int | None = None,
    ) -> None:
        self.lambda_ = lambda_
        self.lambda_nodewise = lambda_nodewise
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.offset = offset
        self.max_iter = max_iter
        self.tol = tol
        self.n_jobs = n_jobs

    def fit(self, X: Any, y: Any) -> "DebiasedPoissonLassoRegressor":
        res = debiased_poisson_lasso(
            X, y,
            lambda_=self.lambda_,
            lambda_nodewise=self.lambda_nodewise,
            alpha=self.alpha,
            fit_intercept=self.fit_intercept,
            standardize=self.standardize,
            offset=self.offset,
            max_iter=self.max_iter,
            tol=self.tol,
            n_jobs=self.n_jobs,
        )
        self._store_result(res)
        return self

    def predict(self, X: Any) -> NDArray[np.float64]:
        """μ̂ = exp(η̂) — matches PoissonLassoRegressor's convention."""
        eta = self.decision_function(X)
        if self.offset is not None:
            eta = eta + np.ascontiguousarray(self.offset, dtype=np.float64)
        return np.exp(eta)


# --- Cox PH ---------------------------------------------------------


@dataclass
class DebiasedCoxResult:
    """Output of :func:`debiased_cox_lasso`.

    Same inferential fields as :class:`DebiasedGLMResult` minus
    ``mu_fitted`` (Cox PH has no ``μ`` in the canonical GLM sense)
    plus ``risk_score`` ``= X β̂`` — the prognostic index on the
    original feature scale.

    Cox has no intercept (the baseline hazard absorbs constants);
    the ``intercept_`` field is the destandardization correction
    ``-⟨β̂_d, x̄⟩`` so that ``X @ coef_ + intercept_`` reproduces the
    linear predictor of the original-feature-scale model.

    Attributes
    ----------
    coef_debiased : ndarray (p,)
    coef_glm : ndarray (p,)
        Underlying penalized Cox lasso fit on the original feature
        scale.
    intercept_ : float
        Destandardization correction. Cox `decision_function` returns
        ``X @ coef_ + intercept_`` — the prognostic index.
    se : ndarray (p,)
    ci_lower, ci_upper, pvalues, z_scores : ndarray (p,)
    risk_score : ndarray (n,)
        ``η̂_i = ⟨x_i, β̂⟩`` at the penalized fit. The Cox prognostic
        index for the training rows.
    Theta : ndarray (p, p)
        Approximate inverse Fisher on the standardized scale, built
        nodewise on the weighted design ``X̃ = W^{1/2} X``.
    lambda_main : float
    lambda_nodewise : ndarray (p,)
    alpha : float
    family : str
        Always ``"cox"``.
    ties : str
        Tie-handling rule used in the partial likelihood.
    """

    coef_debiased: NDArray[np.float64]
    coef_glm: NDArray[np.float64]
    intercept_: float
    se: NDArray[np.float64]
    ci_lower: NDArray[np.float64]
    ci_upper: NDArray[np.float64]
    pvalues: NDArray[np.float64]
    z_scores: NDArray[np.float64]
    risk_score: NDArray[np.float64]
    Theta: NDArray[np.float64]
    lambda_main: float
    lambda_nodewise: NDArray[np.float64]
    alpha: float
    family: str
    ties: str


def debiased_cox_lasso(
    X: Any,
    time: Any,
    event: Any,
    *,
    lambda_: float | None = None,
    lambda_nodewise: float | NDArray[np.float64] | None = None,
    alpha: float = 0.05,
    ties: str = "breslow",
    standardize: bool = True,
    max_iter: int = 1000,
    tol: float = 1e-7,
    n_jobs: int | None = None,
) -> DebiasedCoxResult:
    """Debiased Cox lasso with Wald confidence intervals.

    Construction (van de Geer–Cai–Wang extension of VBR to Cox):

    1. Fit a penalized Cox lasso to obtain ``β̂``.
    2. At ``β̂``, extract the per-sample partial-likelihood Fisher
       information ``w_i`` and working response ``z_i`` from the Cox
       IRLS surrogate (via the Rust ``CoxPH::surrogate_at`` exposed as
       ``_core.cox_surrogate_weights_at``).
    3. Build the **weighted design** ``X̃ = W^{1/2} X`` where
       ``W = diag(w)``. The partial-likelihood Fisher information is
       ``Xᵀ W X``, so ``X̃`` plays exactly the role of ``X`` in the
       LS / GLM debiased construction.
    4. Build ``Θ̂`` by node-wise lasso on ``X̃`` (Van de Geer 2014 §3
       construction; ``_fit_nodewise_column`` is penalty-axis agnostic
       and reused unchanged).
    5. Debias: ``β̂_d = β̂ + (1/n) Θ̂ · Xᵀ (event − μ̂_cox)`` where
       ``μ̂_cox_i = exp(η̂_i) · Λ̂_0(t_i)`` is the partial-likelihood
       contribution. Computed via ``μ̂_cox = event + w·(η − z)`` from
       the surrogate identity ``z = η − g / w``, ``g = event −
       exp(η)·Λ̂_0``.
    6. Variance: ``diag(Θ̂ X̃ᵀX̃ Θ̂ᵀ) / n²``. Same formula as the
       logistic / Poisson case; the partial likelihood supplies its
       own Fisher information, so there is no σ² nuisance term.

    Parameters
    ----------
    X : array-like of shape (n, p)
    time : array-like of shape (n,)
        Event / censoring time; must be ≥ 0.
    event : array-like of shape (n,)
        0/1 censoring indicator (1 = event observed).
    lambda_ : float, optional
        Lasso λ for the penalized Cox fit. Defaults to the
        theoretical ``c · √(log p / n)`` rate.
    lambda_nodewise : float or array, optional
        Per-node λ for the nodewise lassos. Defaults to a tuned grid.
    alpha : float, default 0.05
        Wald CI level — outputs ``[α/2, 1 − α/2]`` percentile bands.
    ties : {"breslow", "efron"}, default "breslow"
        Tie-handling rule for the partial likelihood. Forwarded to
        both the penalized fit and the Fisher-information extraction.
    standardize : bool, default True
    max_iter, tol : solver controls (forwarded to nodewise lassos
        and the penalized Cox fit).
    n_jobs : parallel jobs for the nodewise sweep.

    Returns
    -------
    :class:`DebiasedCoxResult`

    References
    ----------
    van de Geer, S., Bühlmann, P., Ritov, Y., & Dezeure, R. (2014).
    "On asymptotically optimal confidence regions and tests for
    high-dimensional models." *Annals of Statistics* 42(3): 1166–1202.
    — The original LS / GLM debiased construction.

    Cai, T., & Wang, J. (2017). "High-dimensional inference for
    covariance and precision matrices and applications to survival
    analysis." *J. Am. Stat. Assoc.* — Cox-PH extension.
    """
    from joblib import Parallel, delayed
    from skein_glm._core import cox_surrogate_weights_at

    X = np.ascontiguousarray(X, dtype=np.float64)
    time = np.ascontiguousarray(time, dtype=np.float64)
    event = np.ascontiguousarray(event, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    n, p = X.shape
    if time.shape != (n,) or event.shape != (n,):
        raise ValueError(
            f"time / event must be 1D with length {n}; got "
            f"{time.shape} / {event.shape}"
        )
    if not np.all(np.isfinite(time)) or np.any(time < 0):
        raise ValueError("Cox PH requires time ≥ 0 (finite)")
    valid_events = np.isin(event, [0.0, 1.0])
    if not np.all(valid_events):
        raise ValueError("Cox PH requires event ∈ {0, 1}")
    if int(event.sum()) == 0:
        raise ValueError("Cox PH requires at least one event (event = 1)")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    if ties not in ("breslow", "efron"):
        raise ValueError(f"ties must be 'breslow' or 'efron'; got {ties!r}")
    if p < 2:
        raise ValueError(f"debiased lasso requires p ≥ 2 features; got p = {p}")

    Xs, x_mean, x_scale = _standardize_X(X, standardize)
    lambda_main = (
        _theoretical_lambda(n, p) if lambda_ is None else float(lambda_)
    )
    if lambda_main <= 0:
        raise ValueError(f"lambda_ must be > 0; got {lambda_main}")

    # Cox-lasso via MCP at large γ (the standard skein idiom — group-MCP
    # with γ→∞ recovers L1 exactly, and the Rust path supports it
    # natively). Cox has no intercept; the baseline hazard absorbs
    # additive constants.
    fit = CoxMCPRegressor(
        lambda_=lambda_main,
        gamma=1e10,
        ties=ties,
        standardize=False,
        max_iter=max_iter,
        tol=tol,
        max_outer=20,
        outer_tol=tol,
    ).fit(Xs, time, event)
    beta_hat_s = np.asarray(fit.coef_, dtype=np.float64)
    eta_hat = Xs @ beta_hat_s

    # Extract per-sample Cox Fisher information w_i and working
    # response z_i at the fitted β̂. The Rust surrogate returns
    # (w, z) such that z = η − g/w, g = event − exp(η)·Λ̂_0(t).
    w, z = cox_surrogate_weights_at(Xs, time, event, beta_hat_s, ties=ties)
    w = np.asarray(w, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    # μ̂_cox = exp(η)·Λ̂_0(t) = event − g = event + w·(η − z).
    # The GLM core consumes (y, mu_hat) and forms (y − mu_hat) =
    # event − μ̂_cox = -g, the Cox score residual.
    mu_hat_cox = event + w * (eta_hat - z)

    lam_nw = _build_lambda_nodewise(lambda_nodewise, n, p)
    if np.any(lam_nw <= 0):
        raise ValueError("lambda_nodewise entries must be > 0")

    # Inline the GLM-core debiasing math; intercept_s=0 because Cox
    # has none. Mirrors _debiased_glm_core but produces DebiasedCoxResult.
    w_safe = np.maximum(w, _GLM_WEIGHT_FLOOR)
    w_sqrt = np.sqrt(w_safe)
    X_tilde = Xs * w_sqrt[:, None]

    results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_nodewise_column)(X_tilde, j, float(lam_nw[j]), max_iter, tol)
        for j in range(p)
    )
    gammas = [g for g, _ in results]
    tau2s = np.array([t for _, t in results], dtype=np.float64)
    Theta = _assemble_theta_rows(gammas, tau2s)

    score = Xs.T @ (event - mu_hat_cox)
    beta_d_s = beta_hat_s + (Theta @ score) / n

    U = X_tilde @ Theta.T
    var_d_s = np.einsum("ij,ij->j", U, U) / (n * n)
    se_s = np.sqrt(np.maximum(var_d_s, 0.0))

    beta_d = beta_d_s / x_scale
    beta_glm = beta_hat_s / x_scale
    se = se_s / x_scale
    intercept = -float(np.dot(beta_d, x_mean))
    # Risk score is the η̂ on the *original* feature scale —
    # X @ beta_d + intercept reconstructs the centered linear
    # predictor; the Cox partial likelihood is invariant to the
    # additive intercept so this matches the conventional usage.
    risk_score = X @ beta_d + intercept

    z_alpha = float(stats.norm.ppf(1.0 - alpha / 2.0))
    safe_se = np.where(se > 0, se, 1.0)
    z_scores = np.where(se > 0, beta_d / safe_se, 0.0)
    pvalues = 2.0 * stats.norm.sf(np.abs(z_scores))
    ci_lower = beta_d - z_alpha * se
    ci_upper = beta_d + z_alpha * se

    return DebiasedCoxResult(
        coef_debiased=beta_d,
        coef_glm=beta_glm,
        intercept_=intercept,
        se=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        pvalues=pvalues,
        z_scores=z_scores,
        risk_score=risk_score,
        Theta=Theta,
        lambda_main=lambda_main,
        lambda_nodewise=lam_nw,
        alpha=alpha,
        family="cox",
        ties=ties,
    )


class DebiasedCoxLassoRegressor(BaseEstimator):
    """Sklearn-style facade over :func:`debiased_cox_lasso`.

    Cox PH has a different signature from the scalar-``y`` GLMs
    (requires ``time`` and ``event`` instead of ``y``), so this
    estimator doesn't inherit from :class:`_DebiasedGLMRegressorBase`.

    Attributes mirror the other debiased estimators with the GLM
    suffix convention (``coef_``, ``se_``, ``pvalues_``, etc.). The
    Cox-specific :attr:`risk_score_` is set to the prognostic index
    on the training data after :meth:`fit`.
    """

    coef_: NDArray[np.float64]
    coef_glm_: NDArray[np.float64]
    intercept_: float
    se_: NDArray[np.float64]
    ci_lower_: NDArray[np.float64]
    ci_upper_: NDArray[np.float64]
    pvalues_: NDArray[np.float64]
    z_scores_: NDArray[np.float64]
    risk_score_: NDArray[np.float64]
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
        ties: str = "breslow",
        standardize: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-7,
        n_jobs: int | None = None,
    ) -> None:
        self.lambda_ = lambda_
        self.lambda_nodewise = lambda_nodewise
        self.alpha = alpha
        self.ties = ties
        self.standardize = standardize
        self.max_iter = max_iter
        self.tol = tol
        self.n_jobs = n_jobs

    def fit(
        self, X: Any, time: Any, event: Any
    ) -> "DebiasedCoxLassoRegressor":
        res = debiased_cox_lasso(
            X, time, event,
            lambda_=self.lambda_,
            lambda_nodewise=self.lambda_nodewise,
            alpha=self.alpha,
            ties=self.ties,
            standardize=self.standardize,
            max_iter=self.max_iter,
            tol=self.tol,
            n_jobs=self.n_jobs,
        )
        self.coef_ = res.coef_debiased
        self.coef_glm_ = res.coef_glm
        self.intercept_ = res.intercept_
        self.se_ = res.se
        self.ci_lower_ = res.ci_lower
        self.ci_upper_ = res.ci_upper
        self.pvalues_ = res.pvalues
        self.z_scores_ = res.z_scores
        self.risk_score_ = res.risk_score
        self.Theta_ = res.Theta
        self.lambda_main_ = res.lambda_main
        self.lambda_nodewise_ = res.lambda_nodewise
        self.family_ = res.family
        self.ties_ = res.ties
        self.n_features_in_ = int(res.coef_debiased.shape[0])
        return self

    def decision_function(self, X: Any) -> NDArray[np.float64]:
        """Prognostic index ``η = X · β + intercept``."""
        X = np.ascontiguousarray(X, dtype=np.float64)
        return X @ self.coef_ + self.intercept_

    def predict(self, X: Any) -> NDArray[np.float64]:
        """Alias for :meth:`decision_function` (Cox has no μ)."""
        return self.decision_function(X)


__all__ = [
    "DebiasedLassoResult",
    "debiased_lasso",
    "DebiasedLassoRegressor",
    "DebiasedGLMResult",
    "debiased_logistic_lasso",
    "debiased_poisson_lasso",
    "DebiasedLogisticLassoRegressor",
    "DebiasedPoissonLassoRegressor",
    "DebiasedCoxResult",
    "debiased_cox_lasso",
    "DebiasedCoxLassoRegressor",
]
