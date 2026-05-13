"""Graphical-model simulator: sparse precision matrix Ω with known support.

Three topologies:
  - banded:  Ω is k-banded (zeros outside |i-j| ≤ k)
  - hub:     `n_hubs` hub nodes each connected to `hub_degree` peripheral nodes
  - random:  random sparse with `edge_prob` per upper-triangle entry

After choosing Σ = Ω⁻¹, we draw n observations from N(0, Σ).

The Problem returned packages this for the v2 cell driver:
  - problem.x : the n × p sample matrix
  - problem.y : empty (not used for graphical models)
  - problem.beta_true : Ω flattened to p² for the recovery metrics
  - problem.meta["omega_true"] : Ω itself as a p×p ndarray
  - problem.meta["support_mask"] : boolean p×p mask of true edges
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from benches.problems import Problem


Topology = Literal["banded", "hub", "random"]


def _banded_omega(p: int, bandwidth: int = 2, weight: float = 0.4) -> np.ndarray:
    omega = np.eye(p)
    for k in range(1, bandwidth + 1):
        for i in range(p - k):
            omega[i, i + k] = weight ** k
            omega[i + k, i] = weight ** k
    return omega


def _hub_omega(
    p: int, *, n_hubs: int = 3, hub_degree: int = 5,
    weight: float = 0.3, rng: np.random.Generator,
) -> np.ndarray:
    omega = np.eye(p)
    hubs = rng.choice(p, size=n_hubs, replace=False)
    for h in hubs:
        peers = [j for j in range(p) if j != h]
        chosen = rng.choice(peers, size=min(hub_degree, len(peers)), replace=False)
        for j in chosen:
            omega[h, j] = weight
            omega[j, h] = weight
    return omega


def _random_omega(
    p: int, *, edge_prob: float = 0.1, weight: float = 0.3,
    rng: np.random.Generator,
) -> np.ndarray:
    omega = np.eye(p)
    upper = rng.uniform(size=(p, p)) < edge_prob
    upper = np.triu(upper, k=1)
    sign = rng.choice([-1.0, 1.0], size=(p, p))
    weights = upper * weight * sign
    omega = omega + weights + weights.T
    return omega


def _make_psd(omega: np.ndarray, ridge: float = 0.1) -> np.ndarray:
    """Ensure Ω is positive-definite by adding ridge*I until eigenvalues are positive."""
    omega = omega.copy()
    while True:
        try:
            np.linalg.cholesky(omega)
            return omega
        except np.linalg.LinAlgError:
            omega = omega + ridge * np.eye(omega.shape[0])
            ridge *= 2.0


def make(
    *,
    n: int, p: int,
    seed: int = 0,
    topology: Topology = "banded",
    bandwidth: int = 2,
    n_hubs: int = 3,
    hub_degree: int = 5,
    edge_prob: float = 0.05,
    edge_weight: float = 0.3,
) -> Problem:
    rng = np.random.default_rng(seed)
    if topology == "banded":
        omega = _banded_omega(p, bandwidth=bandwidth, weight=edge_weight)
    elif topology == "hub":
        omega = _hub_omega(p, n_hubs=n_hubs, hub_degree=hub_degree,
                           weight=edge_weight, rng=rng)
    elif topology == "random":
        omega = _random_omega(p, edge_prob=edge_prob, weight=edge_weight, rng=rng)
    else:
        raise ValueError(f"unknown topology: {topology!r}")
    omega = _make_psd(omega)
    sigma = np.linalg.inv(omega)
    # Draw n samples from N(0, Σ).
    L = np.linalg.cholesky(sigma)
    z = rng.standard_normal((n, p))
    x = z @ L.T
    support = (np.abs(omega) > 1e-10) & ~np.eye(p, dtype=bool)
    return Problem(
        x=x, y=np.zeros(n),               # y unused
        beta_true=omega.ravel(),
        family="gaussian",                # closest family label for the v2 dispatch
        meta={
            "simulator": "glasso_truth",
            "topology": topology,
            "omega_true": omega,
            "sigma_true": sigma,
            "support_mask": support,
            "n_edges_true": int(support.sum() // 2),
            "p": p,
        },
    )
