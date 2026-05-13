"""Poisson-regression simulator. Keeps μ = exp(η) in a moderate range
so the response isn't pinned at 0 or astronomically large."""
from __future__ import annotations

import math

import numpy as np

from benches.problems import Problem
from benches.v2.simulators._design import design, sparse_beta


def make(
    *,
    n: int, p: int,
    seed: int = 0,
    signal_scale: float = 0.3,
    sparsity_k: float = 1.0,
    corr_kind: str = "iid",
    corr_rho: float = 0.0,
) -> Problem:
    rng = np.random.default_rng(seed)
    x = design(n, p, rng=rng, kind=corr_kind, rho=corr_rho)
    k_active = max(1, int(round(sparsity_k * math.sqrt(p))))
    beta = signal_scale * sparse_beta(p, rng=rng, k_active=k_active)
    eta = x @ beta
    # Clip eta so exp(eta) doesn't overflow at large p; this matters mostly
    # for the "deep" λ-grid where active set saturates.
    eta = np.clip(eta, -10.0, 10.0)
    mu = np.exp(eta)
    y = rng.poisson(mu).astype(np.float64)
    return Problem(
        x=x, y=y, beta_true=beta, family="poisson",
        meta={
            "simulator": "poisson_truth",
            "k_active": k_active, "signal_scale": signal_scale,
            "corr_kind": corr_kind, "corr_rho": corr_rho,
            "mean_count": float(y.mean()),
            "max_count": float(y.max()),
        },
    )
