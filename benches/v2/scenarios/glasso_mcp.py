"""Graphical-MCP scenario. Replaces the L1 envelope in `glasso_l1` with
the nonconvex MCP penalty on Σ⁻¹; the underlying sparse-precision
simulator is unchanged.
"""
from __future__ import annotations

from benches.v2.simulators import glasso_truth

SPEC = {
    "datafit": "gaussian_inv_cov",
    "penalty": "glasso_mcp",
    "family_module": "benches.v2.simulators.glasso_truth",
}


def make_problem_explicit(n: int, p: int, seed: int):
    return glasso_truth.make(n=n, p=p, seed=seed, topology="banded")
