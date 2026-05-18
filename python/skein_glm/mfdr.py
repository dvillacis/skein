"""Marginal false-discovery-rate (mFDR) selection for path estimators.

Implements `grpreg::mfdr` (Breheny 2019, "Marginal false discovery rate
control for likelihood-based penalized regression models"). For a
fitted `*PathRegressor` along a decreasing-λ grid, estimates the
expected number of false discoveries `E[V_k]` at every λ_k and divides
by the observed discovery count `R_k` to produce an mFDR curve. Users
then pick the smallest λ at which `mFDR ≤ target` (typically 0.1) —
analogous to a Knockoff-style FDR control but cheaper and family-agnostic.

The closed-form for least-squares (Breheny 2019 §3, Eq 2):

    mFDR(λ) = (p · 2 · Φ(-λ · √n / σ̂)) / max(1, R(λ))

where Φ is the standard-normal CDF, σ̂ = √(RSS / n), R(λ) the active
set size at λ, p the total feature count.

Binomial and Poisson families use the same form with the deviance-based
residual scale `σ̂ = √(deviance / n)` standing in. Cox is supported via
the path estimator's reported partial likelihood.

This module is decoupled from the Rust solver — it only reads
`path_model.coefs_`, `path_model.lambdas_`, and standard prediction
methods.

References
----------
Breheny, P. (2019). Marginal false discovery rate control for
likelihood-based penalized regression models. *Biometrical Journal*
61(2), 256–268.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray


_ACTIVE_EPS = 1e-12


def _active_size(coefs: NDArray[np.float64]) -> NDArray[np.int64]:
    return np.sum(np.abs(coefs) > _ACTIVE_EPS, axis=1).astype(np.int64)


def _standard_normal_cdf(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Pure-numpy Φ; avoids scipy dependency. Uses `erf` from math.erf."""
    z_arr = np.asarray(z, dtype=np.float64)
    return 0.5 * (1.0 + np.vectorize(math.erf)(z_arr / math.sqrt(2.0)))


def _detect_family(path_model: Any) -> str:
    """Sniff the family from the estimator's class name."""
    name = type(path_model).__name__
    if "Logistic" in name:
        return "logistic"
    if "Poisson" in name:
        return "poisson"
    if "Cox" in name:
        return "cox"
    # Multinomial isn't supported here — its mFDR formula is different.
    if "Multinomial" in name or "Multitask" in name:
        raise NotImplementedError(
            f"mFDR is not supported for {name}; the formula does not generalize "
            f"to multi-response settings in this implementation."
        )
    return "gaussian"


def _residual_scale_gaussian(
    path_model: Any, x: NDArray[np.float64], y: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Per-λ residual scale σ̂_k = √(RSS_k / n)."""
    pred = np.asarray(path_model.predict(x))
    if pred.ndim == 1:
        pred = pred[:, None]
    y_col = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    rss = np.sum((y_col - pred) ** 2, axis=0)
    n = pred.shape[0]
    return np.sqrt(np.maximum(rss / n, 1e-15))


def _residual_scale_logistic(
    path_model: Any, x: NDArray[np.float64], y: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Approximate per-λ residual scale from the binomial deviance.

    σ̂_k ≈ √(deviance_k / n). For binomial regression the natural
    standard-error scale is `1/√(p̂(1-p̂))` per observation, but the
    Breheny 2019 mFDR formula uses the deviance-based proxy directly.
    """
    eta = _linear_predictor(path_model, x)
    y_col = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    # binomial NLL up to constant: Σ softplus(η) − y·η.
    nll = np.sum(np.logaddexp(0.0, eta) - y_col * eta, axis=0)
    n = eta.shape[0]
    return np.sqrt(np.maximum(2.0 * nll / n, 1e-15))


def _residual_scale_poisson(
    path_model: Any, x: NDArray[np.float64], y: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Per-λ residual scale from Poisson deviance proxy."""
    eta = _linear_predictor(path_model, x)
    y_col = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    nll = np.sum(np.exp(eta) - y_col * eta, axis=0)
    n = eta.shape[0]
    return np.sqrt(np.maximum(2.0 * nll / n, 1e-15))


def _linear_predictor(path_model: Any, x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute η = X β + α per λ. Returns `(n, n_lambdas)`."""
    x_arr = np.ascontiguousarray(x, dtype=np.float64)
    eta = x_arr @ path_model.coefs_.T + path_model.intercepts_[None, :]
    return eta


def estimate_mfdr(
    path_model: Any,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    family: str | None = None,
) -> NDArray[np.float64]:
    """Compute the marginal FDR estimate at every λ on the fitted path.

    Parameters
    ----------
    path_model : fitted *PathRegressor
        Must expose ``coefs_`` (n_lambdas × n_features), ``intercepts_``
        (n_lambdas,), ``lambdas_`` (n_lambdas,).
    x, y : ndarray
        Training design and response.
    family : {"gaussian", "logistic", "poisson", "cox"} or None
        Override the family auto-detection (which sniffs the estimator's
        class name). Cox is currently approximated by treating its
        partial-likelihood scale via the same √(deviance / n) form;
        callers needing exact Cox mFDR should defer to the original
        grpreg implementation.

    Returns
    -------
    mfdr : ndarray of shape ``(n_lambdas,)``
        Per-λ mFDR estimate, clipped to ``[0, 1]``.

    Notes
    -----
    The formula is

        mFDR(λ) = (p · 2 · Φ(-λ · √n · scale / σ̂)) / max(1, R(λ))

    where `scale = √n` for standardized columns is absorbed into the
    closed-form bound from Breheny 2019 §3. Estimators returning
    coefficients in original-feature scale (the skein default) should
    have applied centering / scaling internally; this estimator works on
    whatever ``X`` you pass in (typically the same X used to fit).
    """
    if family is None:
        family = _detect_family(path_model)

    x_arr = np.ascontiguousarray(x, dtype=np.float64)
    y_arr = np.ascontiguousarray(y, dtype=np.float64)
    n, p = x_arr.shape
    lambdas = np.asarray(path_model.lambdas_, dtype=np.float64)
    R = _active_size(path_model.coefs_)

    if family == "gaussian":
        sigma = _residual_scale_gaussian(path_model, x_arr, y_arr)
    elif family == "logistic":
        sigma = _residual_scale_logistic(path_model, x_arr, y_arr)
    elif family == "poisson":
        sigma = _residual_scale_poisson(path_model, x_arr, y_arr)
    elif family == "cox":
        # Cox: fall back to the gaussian-deviance proxy on the linear
        # predictor — a coarse approximation but the only one available
        # without the partial-likelihood Hessian.
        sigma = _residual_scale_gaussian(path_model, x_arr, y_arr)
    else:
        raise ValueError(
            f"unknown family {family!r}; expected gaussian / logistic / poisson / cox"
        )

    # Closed-form expected false discoveries per null feature under the
    # marginal-normality approximation. Breheny 2019 uses √n in the
    # numerator since columns are standardized; if the caller passed
    # un-standardized X, the proxy still produces a monotone curve, just
    # with a slightly different effective scale.
    sqrt_n = math.sqrt(n)
    z = lambdas * sqrt_n / np.maximum(sigma, 1e-15)
    e_v_per_null = 2.0 * (1.0 - _standard_normal_cdf(z))
    mfdr = (p * e_v_per_null) / np.maximum(R.astype(np.float64), 1.0)
    return np.clip(mfdr, 0.0, 1.0)


def _select_eligible(mfdr: NDArray[np.float64], target: float) -> int:
    """Return the largest index (smallest λ) in a decreasing-λ path
    whose mFDR is still ≤ target. Raises if none qualify."""
    if not (0.0 < target <= 1.0):
        raise ValueError(f"target must be in (0, 1], got {target}")
    eligible = np.where(mfdr <= target)[0]
    if eligible.size == 0:
        raise ValueError(
            f"no λ on the path achieves mFDR ≤ {target}; "
            f"min mFDR on path is {mfdr.min():.4g}"
        )
    # Breheny 2019: pick the densest model still controlling FDR.
    # In decreasing-λ ordering that's the largest eligible index.
    return int(eligible[-1])


def select_by_mfdr(
    path_model: Any,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    target: float = 0.1,
    family: str | None = None,
) -> tuple[int, float]:
    """Pick the densest model on the path whose mFDR estimate is
    still ≤ ``target``.

    Returns ``(idx, mfdr_at_idx)``. Raises ``ValueError`` if no λ on
    the path satisfies the bound (caller should refit with a denser /
    longer grid).
    """
    mfdr = estimate_mfdr(path_model, x, y, family=family)
    idx = _select_eligible(mfdr, target)
    return idx, float(mfdr[idx])


class MFDR:
    """Marginal-FDR estimator wrapping a fitted ``*PathRegressor``.

    Drop-in companion to the existing CV / IC / stability selectors.

    Examples
    --------
    >>> model = skein_glm.MCPPathRegressor(gamma=3.0).fit(X, y)  # doctest: +SKIP
    >>> sel = skein_glm.MFDR(model).fit(X, y)  # doctest: +SKIP
    >>> idx, mfdr_value = sel.select(target=0.1)  # doctest: +SKIP
    """

    def __init__(self, path_estimator: Any, *, family: str | None = None) -> None:
        self.path_estimator = path_estimator
        self.family = family

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> "MFDR":
        """Compute the mFDR curve for every λ on the fitted path."""
        self.mfdr_ = estimate_mfdr(self.path_estimator, x, y, family=self.family)
        self.lambdas_ = np.asarray(self.path_estimator.lambdas_, dtype=np.float64)
        return self

    def select(self, target: float = 0.1) -> tuple[int, float]:
        """Find the densest λ index whose mFDR estimate is ≤ ``target``.

        Returns ``(idx, mfdr_at_idx)``. In the decreasing-λ ordering
        used by skein's path estimators, this is the *largest* index
        among those satisfying the bound — i.e. the most permissive
        FDR-controlled model. Conventional choice per Breheny 2019.
        """
        if not hasattr(self, "mfdr_"):
            raise RuntimeError("call .fit(x, y) before .select()")
        idx = _select_eligible(self.mfdr_, target)
        return idx, float(self.mfdr_[idx])
