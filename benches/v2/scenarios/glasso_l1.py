"""Graphical-Lasso (L1 on Σ⁻¹) scenario.

The simulator produces a sparse Ω with a controllable topology; the
sample matrix X then comes from N(0, Ω⁻¹). Cell driver delegates the
fit to a graphical-model runner — Phase C ships the scenario module
+ simulator; the matching skein and sklearn runners for the
graphical family land in Phase D.
"""
from __future__ import annotations

from benches.v2.simulators import glasso_truth

SPEC = {
    "datafit": "gaussian_inv_cov",
    "penalty": "lasso",
    "family_module": "benches.v2.simulators.glasso_truth",
}


def make_problem_explicit(n: int, p: int, seed: int):
    return glasso_truth.make(n=n, p=p, seed=seed, topology="banded")
