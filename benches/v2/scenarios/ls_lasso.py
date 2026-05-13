"""Gaussian-LS Lasso scenario."""
from __future__ import annotations

import numpy as np

from benches.problems import SIZES, Size, gaussian_lasso

SPEC = {
    "datafit": "gaussian",
    "penalty": "lasso",
    "family_module": "benches.problems.gaussian_lasso",
}


def make_problem(size: str, seed: int):
    if size in SIZES:
        sz = SIZES[size]
    else:
        # v2 sizes from config.yaml may not match v1 SIZES; build ad hoc.
        # The cell driver resolves the (n, p) tuple before calling us.
        raise KeyError(size)
    return gaussian_lasso(sz, seed=seed)


def make_problem_explicit(n: int, p: int, seed: int):
    """Build a problem at an arbitrary (n, p), needed for v2's scaling cells."""
    return gaussian_lasso(Size(name="explicit", n=n, p=p), seed=seed)
