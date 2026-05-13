"""Group-LS simulator with block correlation + active-group control."""
from __future__ import annotations

import numpy as np

from benches.problems import Problem
from benches.v2.simulators._design import block_sparse_beta, design


def make(
    *,
    n: int, p: int,
    seed: int = 0,
    group_size: int = 5,
    k_active_groups: int | None = None,
    snr: float = 5.0,
    rho_within: float = 0.5,
    rho_between: float = 0.0,
) -> Problem:
    if p % group_size:
        raise ValueError(f"p={p} not divisible by group_size={group_size}")
    rng = np.random.default_rng(seed)
    n_groups = p // group_size
    groups = np.repeat(np.arange(n_groups, dtype=np.int64), group_size)
    x = design(n, p, rng=rng, kind="block", groups=groups,
               rho_within=rho_within, rho_between=rho_between)
    if k_active_groups is None:
        k_active_groups = max(1, int(round(np.sqrt(n_groups))))
    beta = block_sparse_beta(p, groups, rng=rng, k_active_groups=k_active_groups)
    signal = x @ beta
    noise_scale = (float(np.std(signal)) / snr) if snr > 0 else 0.0
    y = signal + noise_scale * rng.standard_normal(n)
    return Problem(
        x=x, y=y, beta_true=beta,
        groups=groups, family="gaussian",
        meta={
            "simulator": "group_truth",
            "snr": snr, "group_size": group_size, "n_groups": n_groups,
            "k_active_groups": k_active_groups,
            "rho_within": rho_within, "rho_between": rho_between,
        },
    )
