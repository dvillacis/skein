"""Model-selection metrics: information criteria along the path, stability
selection FDR/power, CV λ-rule helpers.

All functions take the coefficient path as a (n_lambdas, p) array and
return either a selected λ-index or a per-λ array of criterion values.
"""
from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from benches.v2.metrics import deviance as dev


def _eta_path(
    coef_path: NDArray[np.float64],
    x: NDArray[np.float64],
    intercept_path: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Return (n_lambdas, n) matrix of linear predictors."""
    eta = coef_path @ x.T
    if intercept_path is not None:
        eta = eta + intercept_path[:, None]
    return eta


def _df(coef_path: NDArray[np.float64], rtol: float = 1e-6) -> NDArray[np.int64]:
    """Effective degrees of freedom = active-set size (glmnet/ncvreg convention)."""
    scale = np.maximum(np.max(np.abs(coef_path), axis=1, keepdims=True), 1e-12)
    active = np.abs(coef_path) > rtol * scale
    return active.sum(axis=1).astype(np.int64)


def aic_path(
    coef_path: NDArray[np.float64],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    family: str,
    *,
    intercept_path: NDArray[np.float64] | None = None,
    event: NDArray[np.int64] | None = None,
) -> NDArray[np.float64]:
    """AIC = deviance + 2 · df along the path."""
    eta = _eta_path(coef_path, x, intercept_path)
    devs = np.array([dev.for_family(family, y, eta[k], event=event)
                     for k in range(eta.shape[0])])
    return devs + 2.0 * _df(coef_path)


def bic_path(
    coef_path: NDArray[np.float64],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    family: str,
    *,
    intercept_path: NDArray[np.float64] | None = None,
    event: NDArray[np.int64] | None = None,
) -> NDArray[np.float64]:
    """BIC = deviance + log(n) · df."""
    eta = _eta_path(coef_path, x, intercept_path)
    n = x.shape[0]
    devs = np.array([dev.for_family(family, y, eta[k], event=event)
                     for k in range(eta.shape[0])])
    return devs + math.log(n) * _df(coef_path)


def ebic_path(
    coef_path: NDArray[np.float64],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    family: str,
    *,
    gamma: float = 0.5,
    intercept_path: NDArray[np.float64] | None = None,
    event: NDArray[np.int64] | None = None,
) -> NDArray[np.float64]:
    """EBIC (Chen & Chen 2008): BIC + 2·γ · log(C(p, df))."""
    n, p = x.shape
    eta = _eta_path(coef_path, x, intercept_path)
    devs = np.array([dev.for_family(family, y, eta[k], event=event)
                     for k in range(eta.shape[0])])
    df = _df(coef_path)
    # log C(p, k) via lgamma to avoid overflow.
    log_binom = (math.lgamma(p + 1)
                 - np.array([math.lgamma(int(k) + 1) for k in df])
                 - np.array([math.lgamma(p - int(k) + 1) for k in df]))
    return devs + math.log(n) * df + 2.0 * gamma * log_binom


def argmin_path(crit: NDArray[np.float64]) -> int:
    """Return λ-index minimizing the criterion (lowest is best)."""
    return int(np.argmin(crit))


def lambda_1se_rule(
    cv_mean: NDArray[np.float64],
    cv_se: NDArray[np.float64],
    lambdas_descending: bool = True,
) -> int:
    """glmnet 1-SE rule: among λs whose CV error is within 1 SE of the
    minimum, pick the largest λ (most parsimonious model).

    Conventions assume `lambdas_descending` (the standard glmnet ordering:
    grid runs from λ_max down to λ_min).
    """
    k_min = int(np.argmin(cv_mean))
    thresh = cv_mean[k_min] + cv_se[k_min]
    if lambdas_descending:
        # Indices ≤ k_min correspond to *larger* λs.
        candidates = np.where(cv_mean[: k_min + 1] <= thresh)[0]
        return int(candidates.min()) if candidates.size else k_min
    else:
        candidates = np.where(cv_mean[k_min:] <= thresh)[0]
        return int(k_min + candidates.max()) if candidates.size else k_min


def stability_fdr_power(
    selection_probabilities: NDArray[np.float64],
    true_support: NDArray[np.bool_],
    threshold: float = 0.6,
) -> dict[str, float]:
    """Given per-feature stability-selection probabilities, compute the
    empirical FDR and power at a given threshold.

    selection_probabilities: shape (p,) with values in [0, 1].
    true_support: shape (p,) boolean.
    """
    selected = selection_probabilities >= threshold
    tp = int(np.sum(selected & true_support))
    fp = int(np.sum(selected & ~true_support))
    fn = int(np.sum(~selected & true_support))
    fdr = fp / (tp + fp) if (tp + fp) else 0.0
    power = tp / (tp + fn) if (tp + fn) else 0.0
    return {"fdr": fdr, "power": power, "threshold": threshold,
            "n_selected": int(selected.sum())}


def ic_selection_accuracy(
    coef_path: NDArray[np.float64],
    beta_true: NDArray[np.float64],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    family: str,
    *,
    event: NDArray[np.int64] | None = None,
    rtol: float = 1e-6,
) -> dict[str, dict]:
    """Run AIC/BIC/EBIC on the path and report support-recovery quality
    at each selected λ.

    Returns a dict keyed by criterion → {lambda_index, support_f1,
    beta_rmse}.
    """
    from benches.v2.metrics.recovery import beta_rmse, support_metrics

    aic = aic_path(coef_path, x, y, family, event=event)
    bic = bic_path(coef_path, x, y, family, event=event)
    ebic = ebic_path(coef_path, x, y, family, event=event)

    out: dict[str, dict] = {}
    for name, crit in [("aic", aic), ("bic", bic), ("ebic", ebic)]:
        k = argmin_path(crit)
        bh = coef_path[k]
        sm = support_metrics(bh, beta_true, rtol)
        out[name] = {
            "lambda_index":  k,
            "criterion":     float(crit[k]),
            "support_f1":    sm["support_f1"],
            "support_precision": sm["support_precision"],
            "support_recall": sm["support_recall"],
            "beta_rmse":     beta_rmse(bh, beta_true),
            "hat_support_size": sm["hat_support_size"],
            "true_support_size": sm["true_support_size"],
        }
    return out
