"""Per-family deviance — the GLM goodness-of-fit scale.

Used both for accuracy reporting (does penalized β predict as well as
the unpenalized MLE?) and as the IC-selection numerator
(BIC = deviance + log(n) · df).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def gaussian(y: NDArray[np.float64], eta: NDArray[np.float64],
             sigma2: float | None = None) -> float:
    """Gaussian deviance: ‖y - μ‖² / σ². If σ² is unknown, return SSE."""
    resid = y - eta
    sse = float(np.sum(resid * resid))
    if sigma2 is None or sigma2 <= 0:
        return sse
    return sse / sigma2


def binomial(y: NDArray[np.float64], eta: NDArray[np.float64]) -> float:
    """Binomial deviance: −2 Σ[y log μ + (1-y) log(1-μ)].

    `eta` is the linear predictor; μ = σ(η).
    """
    # log-sum-exp stable form: log(1 + exp(-η)) and log(1 + exp(η))
    log_mu      = -np.logaddexp(0.0, -eta)        # log σ(η)
    log_one_mu  = -np.logaddexp(0.0, eta)         # log(1 - σ(η))
    return -2.0 * float(np.sum(y * log_mu + (1.0 - y) * log_one_mu))


def poisson(y: NDArray[np.float64], eta: NDArray[np.float64]) -> float:
    """Poisson deviance: 2 Σ[y log(y / μ) − (y − μ)].

    For y = 0 the y·log(y/μ) term is taken as 0.
    """
    mu = np.exp(np.clip(eta, -30, 30))
    mask = y > 0
    term = np.zeros_like(y, dtype=float)
    term[mask] = y[mask] * (np.log(y[mask]) - np.log(mu[mask]))
    return 2.0 * float(np.sum(term - (y - mu)))


def cox_breslow(
    eta: NDArray[np.float64],
    time: NDArray[np.float64],
    status: NDArray[np.int64],
) -> float:
    """Cox partial-likelihood deviance under Breslow's tie-breaking.

    Deviance = −2 · log partial likelihood.

    For each event-time t_k, partial likelihood contributes
        η_k − log Σ_{j ∈ R(t_k)} exp(η_j)
    where R(t_k) = {j : time_j ≥ t_k} is the at-risk set.
    """
    order = np.argsort(time)
    t = time[order]
    s = status[order].astype(bool)
    e = eta[order]

    n = e.size
    # Risk set descends from n→1 as we sweep time ascending.
    # log Σ_{j ≥ i} exp(e_j) computed via log-sum-exp suffix scan.
    e_rev = e[::-1]
    # logsumexp prefix scan in reverse.
    m = np.max(e_rev)
    shifted = np.exp(e_rev - m)
    csum = np.cumsum(shifted)
    log_risk_rev = np.log(csum) + m
    log_risk = log_risk_rev[::-1]

    # Breslow: ties at the same time use the SAME risk set; we just sum
    # event contributions independently.
    contrib = e[s] - log_risk[s]
    return -2.0 * float(np.sum(contrib))


def for_family(
    family: str,
    y: NDArray[np.float64],
    eta: NDArray[np.float64],
    *,
    event: NDArray[np.int64] | None = None,
) -> float:
    if family == "gaussian":
        return gaussian(y, eta)
    if family == "logistic":
        return binomial(y, eta)
    if family == "poisson":
        return poisson(y, eta)
    if family == "cox":
        if event is None:
            raise ValueError("Cox deviance needs event status")
        return cox_breslow(eta, y, np.asarray(event, dtype=np.int64))
    raise ValueError(f"unknown family: {family!r}")
