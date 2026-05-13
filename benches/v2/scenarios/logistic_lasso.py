"""Logistic-Lasso scenario."""
from __future__ import annotations

from benches.v2.simulators import logistic_truth

SPEC = {
    "datafit": "logistic",
    "penalty": "lasso",
    "family_module": "benches.v2.simulators.logistic_truth",
}


def make_problem_explicit(n: int, p: int, seed: int):
    return logistic_truth.make(n=n, p=p, seed=seed)
