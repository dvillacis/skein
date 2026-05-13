"""Gaussian-LS Elastic-Net scenario (α=0.5)."""
from __future__ import annotations

from benches.problems import SIZES, Size, gaussian_lasso

SPEC = {
    "datafit": "gaussian",
    "penalty": "elastic_net",
    "family_module": "benches.problems.gaussian_lasso",
}


def make_problem(size: str, seed: int):
    return gaussian_lasso(SIZES[size], seed=seed)


def make_problem_explicit(n: int, p: int, seed: int):
    return gaussian_lasso(Size(name="explicit", n=n, p=p), seed=seed)
