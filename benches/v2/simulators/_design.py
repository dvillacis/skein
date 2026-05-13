"""Correlation-structured design matrices.

The legacy `benches.problems._design` produces iid standard normal X.
For recovery curves we also need:

  - Toeplitz with `rho ∈ [0, 1)`         — autoregressive correlation
  - Equicorrelation                      — all off-diagonals = ρ
  - Block (within ρ_w, between ρ_b)      — group-structured

Each helper returns X drawn from N(0, Σ) for the chosen Σ. Implementation
uses a Cholesky factor of Σ applied to iid noise — O(n p²) memory in the
Cholesky factor, which is fine up to p ≈ 5000 on M1.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray


CorrKind = Literal["iid", "toeplitz", "equicorr", "block"]


def _toeplitz_chol(p: int, rho: float) -> NDArray[np.float64]:
    if not (0.0 <= rho < 1.0):
        raise ValueError(f"toeplitz rho must be in [0, 1), got {rho}")
    if rho == 0.0:
        return np.eye(p)
    # Σ_ij = ρ^|i-j|.
    idx = np.arange(p)
    sigma = rho ** np.abs(idx[:, None] - idx[None, :])
    return np.linalg.cholesky(sigma)


def _equicorr_chol(p: int, rho: float) -> NDArray[np.float64]:
    if not (-1.0 / (p - 1) < rho < 1.0):
        raise ValueError(f"equicorr rho must give a PSD matrix; got {rho}")
    sigma = np.full((p, p), rho)
    np.fill_diagonal(sigma, 1.0)
    return np.linalg.cholesky(sigma)


def _block_chol(p: int, groups: NDArray[np.int64],
                rho_within: float, rho_between: float) -> NDArray[np.float64]:
    """Block-correlated covariance: within-group ρ_w, between-group ρ_b.

    Σ_ij = 1 if i==j; ρ_w if i,j in same group; ρ_b otherwise.
    """
    if groups.shape[0] != p:
        raise ValueError(f"groups length {groups.shape[0]} != p {p}")
    sigma = np.full((p, p), rho_between)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        sigma[np.ix_(idx, idx)] = rho_within
    np.fill_diagonal(sigma, 1.0)
    return np.linalg.cholesky(sigma)


def design(
    n: int, p: int, *,
    rng: np.random.Generator,
    kind: CorrKind = "iid",
    rho: float = 0.0,
    groups: NDArray[np.int64] | None = None,
    rho_within: float = 0.5,
    rho_between: float = 0.0,
) -> NDArray[np.float64]:
    """Draw an n × p design matrix from N(0, Σ) with the requested structure."""
    z = rng.standard_normal((n, p))
    if kind == "iid":
        return z
    if kind == "toeplitz":
        L = _toeplitz_chol(p, rho)
    elif kind == "equicorr":
        L = _equicorr_chol(p, rho)
    elif kind == "block":
        if groups is None:
            raise ValueError("block design requires groups array")
        L = _block_chol(p, groups, rho_within, rho_between)
    else:
        raise ValueError(f"unknown correlation kind: {kind!r}")
    # z is iid N(0,I); z @ L.T has covariance L @ L.T = Σ.
    return z @ L.T


def sparse_beta(
    p: int, *,
    rng: np.random.Generator,
    k_active: int,
    magnitude_lo: float = 0.5,
    magnitude_hi: float = 2.0,
) -> NDArray[np.float64]:
    """Draw a sparse β with `k_active` nonzero entries at random positions."""
    if k_active > p:
        raise ValueError(f"k_active {k_active} > p {p}")
    beta = np.zeros(p)
    idx = rng.choice(p, size=k_active, replace=False)
    signs = rng.choice([-1.0, 1.0], size=k_active)
    mags = rng.uniform(magnitude_lo, magnitude_hi, size=k_active)
    beta[idx] = signs * mags
    return beta


def block_sparse_beta(
    p: int, groups: NDArray[np.int64], *,
    rng: np.random.Generator,
    k_active_groups: int,
) -> NDArray[np.float64]:
    """Block-sparse β: zero out all but `k_active_groups` groups; active groups
    get iid standard-normal entries within."""
    beta = np.zeros(p)
    g_unique = np.unique(groups)
    if k_active_groups > g_unique.size:
        raise ValueError(f"k_active_groups {k_active_groups} > n_groups {g_unique.size}")
    active = rng.choice(g_unique, size=k_active_groups, replace=False)
    for g in active:
        idx = np.where(groups == g)[0]
        beta[idx] = rng.standard_normal(idx.size)
    return beta
