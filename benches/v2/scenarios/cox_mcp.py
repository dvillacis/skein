"""Cox-MCP scenario (γ = 3.0). Event status is in problem.meta;
runners that need it pull it from there.
"""
from __future__ import annotations

from benches.v2.simulators import cox_truth

SPEC = {
    "datafit": "cox",
    "penalty": "mcp",
    "family_module": "benches.v2.simulators.cox_truth",
}


def make_problem_explicit(n: int, p: int, seed: int):
    return cox_truth.make(n=n, p=p, seed=seed)
