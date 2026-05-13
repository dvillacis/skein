"""Gaussian-LS Lasso scenario using the v2 truth-aware simulator.

Differs from `ls_lasso.py` (which delegates to benches.problems for
backward parity) by routing through `benches.v2.simulators.linear_truth`,
exposing correlation + SNR + sparsity knobs and surfacing the linear
predictor / true support in `Problem.meta`.
"""
from __future__ import annotations

from benches.v2.simulators import linear_truth

SPEC = {
    "datafit": "gaussian",
    "penalty": "lasso",
    "family_module": "benches.v2.simulators.linear_truth",
}


def make_problem_explicit(n: int, p: int, seed: int):
    return linear_truth.make(n=n, p=p, seed=seed)
