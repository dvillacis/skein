"""Sparse-Group LS MCP scenario (γ = 3.0). Reuses `gaussian_group`'s
problem generator (group structure + group-supported truth) and lets
the sparse-group MCP penalty add within-group sparsity on top of the
between-group selection.
"""
from __future__ import annotations

from benches.problems import SIZES, Size, gaussian_group

SPEC = {
    "datafit": "gaussian",
    "penalty": "sparse_group_mcp",
    "family_module": "benches.problems.gaussian_group",
}


def make_problem(size: str, seed: int):
    return gaussian_group(SIZES[size], seed=seed)


def make_problem_explicit(n: int, p: int, seed: int):
    return gaussian_group(Size(name="explicit", n=n, p=p), seed=seed)
