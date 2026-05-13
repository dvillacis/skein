"""Gaussian-LS simulator with correlation + SNR + sparsity knobs.

Returns the standard Problem dataclass (defined in benches.problems)
so the rest of the v2 stack sees a unified shape.
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
    snr: float = 5.0,
    sparsity_k: float = 1.0,    # k_active = sparsity_k * sqrt(p), rounded
    corr_kind: str = "iid",
    corr_rho: float = 0.0,
) -> Problem:
    rng = np.random.default_rng(seed)
    x = design(n, p, rng=rng, kind=corr_kind, rho=corr_rho)
    k_active = max(1, int(round(sparsity_k * math.sqrt(p))))
    beta = sparse_beta(p, rng=rng, k_active=k_active)
    signal = x @ beta
    noise_scale = (float(np.std(signal)) / snr) if snr > 0 else 0.0
    y = signal + noise_scale * rng.standard_normal(n)
    return Problem(
        x=x, y=y, beta_true=beta,
        family="gaussian",
        meta={
            "simulator": "linear_truth",
            "snr": snr, "k_active": k_active,
            "corr_kind": corr_kind, "corr_rho": corr_rho,
            "noise_scale": noise_scale,
        },
    )
