"""Logistic-regression simulator with controllable signal magnitude.

We scale β so the linear predictor stays in a regime where support is
identifiable but classes aren't fully separable — large β leads to
perfect separation and MLE non-existence.
"""
from __future__ import annotations

import math

import numpy as np

from benches.problems import Problem
from benches.v2.simulators._design import design, sparse_beta


def make(
    *,
    n: int, p: int,
    seed: int = 0,
    signal_scale: float = 1.0,
    sparsity_k: float = 1.0,
    corr_kind: str = "iid",
    corr_rho: float = 0.0,
) -> Problem:
    rng = np.random.default_rng(seed)
    x = design(n, p, rng=rng, kind=corr_kind, rho=corr_rho)
    k_active = max(1, int(round(sparsity_k * math.sqrt(p))))
    beta = signal_scale * sparse_beta(p, rng=rng, k_active=k_active)
    eta = x @ beta
    prob = 1.0 / (1.0 + np.exp(-eta))
    y = (rng.uniform(size=n) < prob).astype(np.float64)
    return Problem(
        x=x, y=y, beta_true=beta, family="logistic",
        meta={
            "simulator": "logistic_truth",
            "k_active": k_active, "signal_scale": signal_scale,
            "corr_kind": corr_kind, "corr_rho": corr_rho,
            "class_balance": float(y.mean()),
        },
    )
