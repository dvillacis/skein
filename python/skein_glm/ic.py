"""Information-criterion λ selection for path estimators.

`select_by_ic(path_model, x, *outcomes, criterion=...)` picks the best
λ from a fitted `*PathRegressor` by minimizing AIC, BIC, or EBIC. It
composes with any existing path estimator without per-class wrappers
(unlike the CV API, where the K-fold loop has to live inside an
estimator); the user simply asks "which λ on this fitted path is the
IC-best one?"

Per-datafit dispatch sniffs the path estimator's class name. The four
GLM families currently supported:

- LS (`MCPPathRegressor`, `SCADPathRegressor`, `*Group*PathRegressor`):
  NLL = `(n/2) · log(RSS/n)` (drops the additive Gaussian constant —
  fine for AIC/BIC since they're additive-constant-invariant).
- Logistic: NLL = `Σ_i softplus(η_i) − y_i · η_i` (binomial NLL up to
  the y log y constant).
- Poisson: NLL = `Σ_i exp(η_i) − y_i · η_i` (drops `log(y_i!)`).
- Cox PH: NLL is read from `path_model.info_["final_losses"]` (Cox
  partial NLL per sample, multiplied by `n` to get the total).

Effective degrees of freedom = the active-set size at each λ
(`np.sum(|β| > eps)`). This matches the unbiased-estimator df result
for lasso (Zou-Hastie-Tibshirani) and is the standard convention in
ncvreg / glmnet.

Three criteria, all minimized:

- AIC: `2k + 2·NLL`
- BIC: `log(n)·k + 2·NLL`
- EBIC (Chen-Chen): `BIC + 2γ·log C(p,k)` with `γ ∈ [0, 1]` (default
  0.5). Tighter than BIC when `p ≫ n`; matches what `ncvreg::BIC`
  uses for high-dim selection.
"""
from __future__ import annotations


import numpy as np
from numpy.typing import NDArray

from skein_glm.estimators import _is_sparse  # noqa: F401  (kept for potential downstream use)


_DEFAULT_ACTIVE_EPS = 1e-12


def _active_size(coefs: NDArray[np.float64], eps: float = _DEFAULT_ACTIVE_EPS) -> NDArray[np.int64]:
    """Per-λ count of non-zero β entries."""
    return np.sum(np.abs(coefs) > eps, axis=1).astype(np.int64)


def _compute_nll_ls(path_model, x, y) -> NDArray[np.float64]:
    """Gaussian-LS NLL up to additive constants: `(n/2) · log(RSS/n)`."""
    pred = np.asarray(path_model.predict(x))  # (n, n_lambdas)
    if pred.ndim == 1:
        pred = pred[:, None]
    y_col = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    rss = np.sum((y_col - pred) ** 2, axis=0)
    n = pred.shape[0]
    return 0.5 * n * np.log(np.maximum(rss / n, 1e-15))


def _compute_nll_logistic(path_model, x, y) -> NDArray[np.float64]:
    """Binomial NLL: `Σ softplus(η) − y·η`. `decision_function` returns
    η directly (path version: shape `(n, n_lambdas)`)."""
    eta = np.asarray(path_model.decision_function(x))
    if eta.ndim == 1:
        eta = eta[:, None]
    # Numerically stable softplus.
    sp = np.maximum(eta, 0.0) + np.log1p(np.exp(-np.abs(eta)))
    y_col = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    return np.sum(sp - y_col * eta, axis=0)


def _compute_nll_poisson(path_model, x, y) -> NDArray[np.float64]:
    """Poisson (log-link) NLL up to `log(y!)`: `Σ exp(η) − y·η`. The
    same η-clamp the Rust core uses (`±30`) keeps `exp(η)` finite."""
    eta = np.asarray(path_model.decision_function(x))
    if eta.ndim == 1:
        eta = eta[:, None]
    eta_clamped = np.clip(eta, -30.0, 30.0)
    mu = np.exp(eta_clamped)
    y_col = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    return np.sum(mu - y_col * eta_clamped, axis=0)


def _compute_nll_multinomial(path_model, x, y) -> NDArray[np.float64]:
    """Multinomial NLL: per-λ Σ_i (logsumexp(η_i) − η_{i, y_i}). The path
    estimator's `coefs_` has shape `(n_lambdas, K, p)` and η is the per-λ
    `(n, K)` matrix from `decision_function` recomputed at each λ — but
    `decision_function` only honors the single-best-λ refit, so we
    recompute η directly off `coefs_` / `intercepts_`."""
    coefs_3d: NDArray[np.float64] = path_model.coefs_  # (n_lambdas, K, p)
    intercepts: NDArray[np.float64] = path_model.intercepts_  # (n_lambdas, K)
    n_lambdas = coefs_3d.shape[0]
    classes = np.asarray(path_model.classes_)
    y_arr = np.asarray(y)
    codes = np.searchsorted(classes, y_arr).astype(np.intp)
    n = codes.shape[0]

    # Build a single (n, K) η per λ.
    nll = np.zeros(n_lambdas)
    rows = np.arange(n)
    for li in range(n_lambdas):
        if hasattr(x, "toarray"):
            eta = (x @ coefs_3d[li].T)
            if hasattr(eta, "toarray"):
                eta = eta.toarray()
            eta = np.asarray(eta) + intercepts[li]
        else:
            eta = np.asarray(x) @ coefs_3d[li].T + intercepts[li]
        m = eta.max(axis=1, keepdims=True)
        lse = (np.log(np.exp(eta - m).sum(axis=1, keepdims=True)) + m).ravel()
        eta_y = eta[rows, codes]
        nll[li] = float(np.sum(lse - eta_y))
    return nll


def _multinomial_active_size(coefs_3d: NDArray[np.float64], eps: float) -> NDArray[np.int64]:
    """Active-feature count per λ for row-grouped multinomial: a feature
    is active if **any** of its K class coefficients is nonzero. `coefs_3d`
    has shape `(n_lambdas, K, p)`."""
    # Take max-abs across the class axis, then count features above eps.
    max_abs_per_feature = np.max(np.abs(coefs_3d), axis=1)  # (n_lambdas, p)
    return np.sum(max_abs_per_feature > eps, axis=1).astype(np.int64)


def _compute_nll_cox(path_model, time) -> NDArray[np.float64]:
    """Cox PH NLL: `path_model.info_["final_losses"][k]` is already the
    Breslow per-sample partial NLL at λ_k (the same expression
    `CoxPH::loss` returns), so multiply by `n` to get the total."""
    losses = np.asarray(path_model.info_["final_losses"], dtype=np.float64)
    n = len(time)
    return losses * n


def _detect_family(path_model) -> str:
    """Sniff the datafit family from the estimator's class name. Cox
    has the unique `(time, event)` outcome shape and so the dispatch
    is part of the contract; logistic/Poisson/LS are picked for the
    NLL helper but otherwise share one path."""
    cls = type(path_model).__name__
    if "Cox" in cls:
        return "cox"
    if "Multinomial" in cls:
        return "multinomial"
    if "Logistic" in cls:
        return "logistic"
    if "Poisson" in cls:
        return "poisson"
    return "ls"


def select_by_ic(
    path_model,
    x,
    *outcomes,
    criterion: str = "bic",
    ebic_gamma: float = 0.5,
    active_eps: float = _DEFAULT_ACTIVE_EPS,
) -> tuple[int, NDArray[np.float64]]:
    """Pick the best λ from a fitted path estimator by AIC, BIC, or EBIC.

    A single free function — no per-estimator wrapper. Dispatches on
    the path estimator's class name to compute the right negative
    log-likelihood (LS, logistic, Poisson, or Cox), then adds a
    complexity penalty:

    - AIC = 2k + 2·NLL
    - BIC = log(n)·k + 2·NLL
    - EBIC = BIC + 2γ·log C(p, k), with γ ∈ [0, 1] (default 0.5;
      matches ``ncvreg::BIC``'s high-dim recommendation)

    where k is the number of nonzero coefficients per λ (the
    Zou-Hastie-Tibshirani unbiased df estimator, the standard
    ``ncvreg`` / ``glmnet`` convention).

    Parameters
    ----------
    path_model : *PathRegressor
        Any fitted path estimator (LS / logistic / Poisson / Cox).
    x : array-like
        The design matrix used in the fit. Used to recompute the
        per-λ negative log-likelihood.
    *outcomes
        For non-Cox estimators: a single ``y`` array. For Cox:
        ``time, event``. Mirrors each estimator's ``fit`` signature.
    criterion : {"aic", "bic", "ebic"}, default "bic"
        Which information criterion to use.
    ebic_gamma : float, default 0.5
        EBIC penalty parameter γ ∈ [0, 1]. Ignored for AIC/BIC.
    active_eps : float, default 1e-12
        Threshold for counting a coefficient as "active" (nonzero).

    Returns
    -------
    best_idx : int
        Index into ``path_model.lambdas_`` of the IC-minimizing λ.
    scores : ndarray of shape (n_lambdas,)
        Per-λ score vector (lower-is-better). The fitted β is
        ``path_model.coefs_[best_idx]``.

    Examples
    --------
    >>> import skein_glm
    >>> path = skein_glm.MCPPathRegressor(gamma=3.0, n_lambdas=50).fit(X, y)
    >>> best_idx, scores = skein_glm.select_by_ic(path, X, y, criterion="bic")
    >>> beta_best = path.coefs_[best_idx]
    >>> intercept_best = path.intercepts_[best_idx]

    For Cox PH:

    >>> cox_path = skein_glm.CoxMCPPathRegressor(gamma=3.0, n_lambdas=50).fit(
    ...     X, time, event)
    >>> best_idx, _ = skein_glm.select_by_ic(cox_path, X, time, event, criterion="ebic")
    """
    if criterion not in ("aic", "bic", "ebic"):
        raise ValueError(
            f"criterion must be 'aic', 'bic', or 'ebic'; got {criterion!r}"
        )
    if not 0.0 <= ebic_gamma <= 1.0:
        raise ValueError(
            f"ebic_gamma must be in [0, 1]; got {ebic_gamma}"
        )

    family = _detect_family(path_model)
    if family == "cox":
        if len(outcomes) != 2:
            raise ValueError(
                "Cox path estimators require `(time, event)` outcomes"
            )
        time, _event = outcomes
        nll = _compute_nll_cox(path_model, time)
        n = len(time)
    else:
        if len(outcomes) != 1:
            raise ValueError(
                f"{family} path estimators take a single `y` outcome"
            )
        y = outcomes[0]
        if family == "logistic":
            nll = _compute_nll_logistic(path_model, x, y)
        elif family == "poisson":
            nll = _compute_nll_poisson(path_model, x, y)
        elif family == "multinomial":
            nll = _compute_nll_multinomial(path_model, x, y)
        else:
            nll = _compute_nll_ls(path_model, x, y)
        n = len(y)

    coefs: NDArray[np.float64] = path_model.coefs_
    if family == "multinomial":
        # coefs_ shape (n_lambdas, K, p); df = active row count.
        p = coefs.shape[2]
        k = _multinomial_active_size(coefs, active_eps).astype(np.float64)
    else:
        p = coefs.shape[1]
        k = _active_size(coefs, active_eps).astype(np.float64)

    if criterion == "aic":
        scores = 2.0 * k + 2.0 * nll
    elif criterion == "bic":
        scores = np.log(n) * k + 2.0 * nll
    else:  # "ebic"
        from scipy.special import gammaln  # type: ignore[import-untyped]
        # log C(p, k) = lgamma(p+1) - lgamma(k+1) - lgamma(p-k+1).
        log_binom = (
            gammaln(p + 1.0) - gammaln(k + 1.0) - gammaln(p - k + 1.0)
        )
        scores = np.log(n) * k + 2.0 * nll + 2.0 * ebic_gamma * log_binom

    best_idx = int(np.argmin(scores))
    return best_idx, scores
