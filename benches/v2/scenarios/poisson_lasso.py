"""Poisson-Lasso scenario."""
from __future__ import annotations

from benches.v2.simulators import poisson_truth

SPEC = {
    "datafit": "poisson",
    "penalty": "lasso",
    "family_module": "benches.v2.simulators.poisson_truth",
}


def make_problem_explicit(n: int, p: int, seed: int):
    return poisson_truth.make(n=n, p=p, seed=seed)
