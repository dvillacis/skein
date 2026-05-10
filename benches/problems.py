"""Synthetic problem generators shared by the M9 bench suite.

Conventions mirror tests/fixtures/generate.R so Python and R benchmarks
operate on byte-identical problems where it matters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray


SizeName = Literal["small", "medium", "large"]


@dataclass(frozen=True)
class Size:
    name: SizeName
    n: int
    p: int


SIZES: dict[SizeName, Size] = {
    "small": Size("small", n=1_000, p=100),
    "medium": Size("medium", n=10_000, p=1_000),
    "large": Size("large", n=100_000, p=10_000),
}


@dataclass(frozen=True)
class Problem:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    beta_true: NDArray[np.float64]
    groups: NDArray[np.int64] | None = None
    family: Literal["gaussian", "logistic", "poisson", "cox"] = "gaussian"
    meta: dict[str, object] = field(default_factory=dict)


def _design(n: int, p: int, rng: np.random.Generator) -> NDArray[np.float64]:
    return rng.standard_normal((n, p))


def _sparse_beta(p: int, k_active: int, rng: np.random.Generator) -> NDArray[np.float64]:
    beta = np.zeros(p)
    idx = rng.choice(p, size=k_active, replace=False)
    beta[idx] = rng.choice([-1.0, 1.0], size=k_active) * rng.uniform(0.5, 2.0, size=k_active)
    return beta


def gaussian_lasso(size: Size, *, k_active: int = 10, snr: float = 5.0, seed: int = 1) -> Problem:
    rng = np.random.default_rng(seed)
    x = _design(size.n, size.p, rng)
    beta = _sparse_beta(size.p, k_active, rng)
    signal = x @ beta
    noise_scale = float(np.std(signal)) / snr
    y = signal + noise_scale * rng.standard_normal(size.n)
    return Problem(x=x, y=y, beta_true=beta, family="gaussian", meta={"snr": snr, "k_active": k_active})


def gaussian_group(
    size: Size, *, group_size: int = 5, k_active_groups: int = 5, snr: float = 5.0, seed: int = 1
) -> Problem:
    if size.p % group_size:
        raise ValueError(f"p={size.p} not divisible by group_size={group_size}")
    rng = np.random.default_rng(seed)
    x = _design(size.n, size.p, rng)
    n_groups = size.p // group_size
    groups = np.repeat(np.arange(n_groups, dtype=np.int64), group_size)
    beta = np.zeros(size.p)
    active = rng.choice(n_groups, size=k_active_groups, replace=False)
    for g in active:
        block = slice(g * group_size, (g + 1) * group_size)
        beta[block] = rng.standard_normal(group_size)
    signal = x @ beta
    noise_scale = float(np.std(signal)) / snr
    y = signal + noise_scale * rng.standard_normal(size.n)
    return Problem(
        x=x,
        y=y,
        beta_true=beta,
        groups=groups,
        family="gaussian",
        meta={"snr": snr, "group_size": group_size, "n_groups": n_groups},
    )


def logistic(size: Size, *, k_active: int = 10, seed: int = 2) -> Problem:
    rng = np.random.default_rng(seed)
    x = _design(size.n, size.p, rng)
    beta = _sparse_beta(size.p, k_active, rng)
    eta = x @ beta
    prob = 1.0 / (1.0 + np.exp(-eta))
    y = (rng.uniform(size=size.n) < prob).astype(np.float64)
    return Problem(x=x, y=y, beta_true=beta, family="logistic", meta={"k_active": k_active})


def poisson(size: Size, *, k_active: int = 10, seed: int = 3) -> Problem:
    rng = np.random.default_rng(seed)
    x = _design(size.n, size.p, rng)
    beta = _sparse_beta(size.p, k_active, rng) * 0.3
    mu = np.exp(x @ beta)
    y = rng.poisson(mu).astype(np.float64)
    return Problem(x=x, y=y, beta_true=beta, family="poisson", meta={"k_active": k_active})
