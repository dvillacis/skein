"""Model selection for graphical models.

The headline export is :func:`ebic_path`, which walks a λ-grid and
selects the precision matrix that minimises the Extended Bayesian
Information Criterion (Foygel & Drton 2010). EBIC is the field-standard
tuning rule in network psychometrics — it penalises edge count more
aggressively than BIC and gives stable graphs even at small `n / p`.

For joint glasso, :func:`joint_ebic_path` walks a `(λ_2)` grid summing
per-population log-likelihoods and counting the *union* of active edges
across populations.

Conventions:

- `gamma ∈ [0, 1]` controls the EBIC strength. `gamma = 0` is plain
  BIC; `gamma = 0.5` is the field default for graphical models.
- `n` is required when the user supplied a precomputed covariance
  (we can't infer it from `S` shape).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class EBICPathResult:
    """Outcome of a single-population EBIC search.

    Attributes
    ----------
    best_estimator : fitted estimator at the EBIC-selected λ.
    best_lambda : float
    best_ebic : float
    lambdas : ndarray (n_lambdas,)
    ebic : ndarray (n_lambdas,)
    n_edges : ndarray (n_lambdas,) int
    """

    best_estimator: Any
    best_lambda: float
    best_ebic: float
    lambdas: NDArray[np.float64]
    ebic: NDArray[np.float64]
    n_edges: NDArray[np.int64]


def _count_edges(theta: NDArray[np.float64], atol: float = 1e-8) -> int:
    """Number of nonzero upper-triangular off-diagonal entries."""
    p = theta.shape[0]
    iu = np.triu_indices(p, k=1)
    return int(np.sum(np.abs(theta[iu]) > atol))


def _gaussian_loglik(s: NDArray[np.float64], theta: NDArray[np.float64], n: float) -> float:
    """Gaussian log-likelihood (up to constants) of `Θ̂` given sample
    covariance `S` and effective sample size `n`:

        ℓ(Θ̂; S) = (n / 2) · (log det Θ̂ − tr(SΘ̂)).
    """
    sign, logdet = np.linalg.slogdet(theta)
    if sign <= 0:
        # Non-PD; return -inf to disqualify.
        return -np.inf
    return 0.5 * n * (logdet - float(np.trace(s @ theta)))


def ebic_path(
    X,
    estimator_cls,
    lambdas: NDArray[np.float64],
    *,
    gamma: float = 0.5,
    n: int | None = None,
    assume_centered: bool = False,
    **estimator_kwargs,
) -> EBICPathResult:
    """Sweep a λ-grid for a graphical estimator and pick the model
    minimising EBIC.

    Parameters
    ----------
    X : ndarray
        Either raw `(n, p)` data or a precomputed `(p, p)` symmetric
        covariance. If precomputed, pass `n` explicitly.
    estimator_cls : a class with an `alpha` parameter (GraphicalLasso,
        GraphicalMCP, GraphicalSCAD).
    lambdas : array-like of floats (positive, ideally in descending order).
    gamma : float, default 0.5
        EBIC strength. `0` → BIC.
    n : int, optional
        Effective sample size. Required if `X` is precomputed
        covariance.
    assume_centered : bool, default False
    **estimator_kwargs : passed through to the estimator constructor.

    Returns
    -------
    EBICPathResult
    """
    from skein_glm.estimators import _is_symmetric, _to_covariance

    x = np.ascontiguousarray(X, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {x.shape}")
    rows, cols = x.shape
    if rows == cols and _is_symmetric(x):
        if n is None:
            raise ValueError(
                "n is required when X is a precomputed covariance matrix"
            )
        s = x
        p = cols
    else:
        s = _to_covariance(x, assume_centered=assume_centered)
        n = rows if n is None else n
        p = s.shape[0]
    assert n is not None

    lambdas = np.ascontiguousarray(lambdas, dtype=np.float64)
    if lambdas.ndim != 1 or lambdas.size == 0:
        raise ValueError("lambdas must be a non-empty 1D array")
    if np.any(lambdas <= 0):
        raise ValueError("lambdas must be positive")

    ebic_values = np.empty(lambdas.size, dtype=np.float64)
    n_edges = np.empty(lambdas.size, dtype=np.int64)
    estimators: list[Any] = []
    log_p = np.log(p) if p > 1 else 0.0
    log_n = np.log(n) if n > 1 else 0.0

    for i, lam in enumerate(lambdas):
        est = estimator_cls(alpha=float(lam), assume_centered=assume_centered, **estimator_kwargs)
        est.fit(s)
        e = _count_edges(est.precision_)
        ll = _gaussian_loglik(s, est.precision_, n)
        ebic_values[i] = -2.0 * ll + e * log_n + 4.0 * gamma * e * log_p
        n_edges[i] = e
        estimators.append(est)

    best_i = int(np.argmin(ebic_values))
    return EBICPathResult(
        best_estimator=estimators[best_i],
        best_lambda=float(lambdas[best_i]),
        best_ebic=float(ebic_values[best_i]),
        lambdas=lambdas,
        ebic=ebic_values,
        n_edges=n_edges,
    )


@dataclass
class JointEBICPathResult:
    best_estimator: Any
    best_lambda_2: float
    best_ebic: float
    lambdas: NDArray[np.float64]
    ebic: NDArray[np.float64]
    n_edges_union: NDArray[np.int64]


def joint_ebic_path(
    Xs: list,
    estimator_cls,
    lambdas: NDArray[np.float64],
    *,
    gamma: float = 0.5,
    ns: list[int] | None = None,
    assume_centered: bool = False,
    **estimator_kwargs,
) -> JointEBICPathResult:
    """EBIC tuner for joint glasso. Walks a `λ_2` grid; the active
    edge count is the *union* across populations (an edge counts once
    if any population has it nonzero).

    Sums per-population log-likelihoods. Per-pop `n_k` is read from
    raw `X^(k)` row count, or via the `ns` argument when populations
    are passed as precomputed covariances.
    """
    from skein_glm.estimators import _is_symmetric, _to_covariance

    if not isinstance(Xs, (list, tuple)) or len(Xs) == 0:
        raise ValueError("Xs must be a non-empty list of arrays")

    covs: list[NDArray[np.float64]] = []
    n_list: list[int] = []
    p: int | None = None
    for k, x in enumerate(Xs):
        xa = np.ascontiguousarray(x, dtype=np.float64)
        if xa.ndim != 2:
            raise ValueError(f"Xs[{k}] must be 2D, got shape {xa.shape}")
        rows, cols = xa.shape
        if rows == cols and _is_symmetric(xa):
            if ns is None:
                raise ValueError(
                    "ns is required when Xs contain precomputed covariance matrices"
                )
            covs.append(xa)
            n_list.append(int(ns[k]))
            this_p = cols
        else:
            covs.append(_to_covariance(xa, assume_centered=assume_centered))
            n_list.append(int(rows))
            this_p = cols
        if p is None:
            p = this_p
        elif this_p != p:
            raise ValueError("populations must share the same number of features")
    assert p is not None

    lambdas = np.ascontiguousarray(lambdas, dtype=np.float64)
    if lambdas.ndim != 1 or lambdas.size == 0:
        raise ValueError("lambdas must be a non-empty 1D array")

    ebic_values = np.empty(lambdas.size, dtype=np.float64)
    n_edges_union = np.empty(lambdas.size, dtype=np.int64)
    estimators: list[Any] = []
    log_p = np.log(p) if p > 1 else 0.0
    n_total = float(sum(n_list))
    log_n_total = np.log(n_total) if n_total > 1 else 0.0

    for i, lam in enumerate(lambdas):
        est = estimator_cls(
            lambda_2=float(lam),
            assume_centered=assume_centered,
            **estimator_kwargs,
        )
        est.fit(Xs)
        # Per-pop log-lik, edge union.
        ll_total = 0.0
        union_mask = np.zeros((p, p), dtype=bool)
        for theta, s, n_k in zip(est.precisions_, covs, n_list):
            ll_total += _gaussian_loglik(s, theta, float(n_k))
            union_mask |= np.abs(theta) > 1e-8
        iu = np.triu_indices(p, k=1)
        e = int(np.sum(union_mask[iu]))
        ebic_values[i] = -2.0 * ll_total + e * log_n_total + 4.0 * gamma * e * log_p
        n_edges_union[i] = e
        estimators.append(est)

    best_i = int(np.argmin(ebic_values))
    return JointEBICPathResult(
        best_estimator=estimators[best_i],
        best_lambda_2=float(lambdas[best_i]),
        best_ebic=float(ebic_values[best_i]),
        lambdas=lambdas,
        ebic=ebic_values,
        n_edges_union=n_edges_union,
    )
